# Copyright (c) 2023 Andy Zou https://github.com/llm-attacks/llm-attacks. All rights reserved.
# This file has been modified by Zijun Wang ("Zijun Wang Modifications").
# All Zijun Wang Modifications are Copyright (C) 2023 Zijun Wang. All rights reserved.

import gc
import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from filelock import FileLock

from AttnGCG import AttackPrompt, MultiPromptAttack, PromptManager
from AttnGCG import get_embedding_matrix, get_embeddings


def token_gradients(model, input_ids, input_slice, target_slice, loss_slice, goal_slice, 
                    control_slice, sys_prompt_slice, assistant_role_slice,
                    target_weight, attention_pooling_method, attention_weight, attention_weight_dict):
    
    assert attention_pooling_method
    assert attention_weight_dict

    embed_weights = get_embedding_matrix(model)
    one_hot = torch.zeros(
        input_ids[input_slice].shape[0],
        embed_weights.shape[0],
        device=model.device,
        dtype=embed_weights.dtype
    )
    one_hot.scatter_(
        1, 
        input_ids[input_slice].unsqueeze(1),
        torch.ones(one_hot.shape[0], 1, device=model.device, dtype=embed_weights.dtype)
    )
    one_hot.requires_grad_()
    input_embeds = (one_hot @ embed_weights).unsqueeze(0)
    
    # now stitch it together with the rest of the embeddings
    embeds = get_embeddings(model, input_ids.unsqueeze(0)).detach()
    full_embeds = torch.cat(
        [
            embeds[:,:input_slice.start,:], 
            input_embeds, 
            embeds[:,input_slice.stop:,:]
        ], 
        dim=1)
    
    x = model(inputs_embeds=full_embeds, output_attentions=True)
    logits = x.logits
    attentions = x.attentions[-1]
    slice_dict = {
        'goal': goal_slice,
        'sys_role': sys_prompt_slice,
        'control': control_slice,
    }
    weight_dict = attention_weight_dict
    
    attn = attentions[:, :, loss_slice.start:].mean(2)
    tmp = attn.mean(1)
    tmp_input = tmp[:, :assistant_role_slice.stop]
    loss2 = torch.zeros(len(tmp_input)).to(attentions.device)
    for name, slices in slice_dict.items():
        if attention_pooling_method=='mean':
            val = tmp_input[0, slices].mean().to(dtype=torch.float32)
        elif attention_pooling_method=='sum':
            val = tmp_input[0, slices].sum().to(dtype=torch.float32)
        else:
            raise ValueError(f"Invalid Attention_pooling_method, expect 'mean' or 'sum', get {attention_pooling_method}") 
        loss2 +=  val * weight_dict[name]
        
    targets = input_ids[target_slice]
    loss1 = nn.CrossEntropyLoss()(logits[0,loss_slice,:], targets)
    loss = target_weight*loss1 + attention_weight*loss2
    
    loss.backward()
    
    del x, logits, attentions, loss, loss1, loss2; gc.collect()
    return one_hot.grad.clone()


class AttnGCGAttackPrompt(AttackPrompt):

    def __init__(self, *args, **kwargs):
        
        super().__init__(*args, **kwargs)
    
    def grad(self, model, mode="control", 
                target_weight=None,
                attention_pooling_method=None, 
                attention_weight=None, 
                attention_weight_dict=None):
        if mode == "control":
            return token_gradients(
                model, 
                self.input_ids.to(model.device), 
                self._control_slice, 
                self._target_slice, 
                self._loss_slice,
                self._goal_slice,
                self._control_slice,
                self._sys_prompt_slice,
                self._assistant_role_slice,
                target_weight,
                attention_pooling_method,
                attention_weight, 
                attention_weight_dict
            )

class AttnGCGPromptManager(PromptManager):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

    def sample_control(self, grad, mode, batch_size, topk=256, temp=1, allow_non_ascii=True):

        if not allow_non_ascii:
            grad[:, self._nonascii_toks.to(grad.device)] = np.infty
        top_indices = (-grad).topk(topk, dim=1).indices
        
        if mode == "control":
            control_toks = self.control_toks.to(grad.device)
        else:
            raise ValueError(f"Invalid mode type, should be 'control', got {mode}")
        
        original_control_toks = control_toks.repeat(batch_size, 1)
        new_token_pos = torch.arange(
            0, 
            len(control_toks), 
            len(control_toks) / batch_size,
            device=grad.device
        ).type(torch.int64)
        new_token_val = torch.gather(
            top_indices[new_token_pos], 1, 
            torch.randint(0, topk, (batch_size, 1),
            device=grad.device)
        )
        new_control_toks = original_control_toks.scatter_(1, new_token_pos.unsqueeze(-1), new_token_val)
        
        return new_control_toks


class AttnGCGMultiPromptAttack(MultiPromptAttack):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)
    
    def step(self, 
             batch_size=1024, 
             topk=256, 
             temp=1, 
             allow_non_ascii=True, 
             target_weight=1.0, 
             control_weight=0.1, 
             verbose=False, 
             attention_pooling_method=None,
             attention_weight=None,
             attention_weight_dict=None,
             enable_prefix_sharing=True,
             merge_loss_compute=True,
             filter_cand=True,
             n_step=0):

        loss_config = []
        if merge_loss_compute:
            loss_config.append((target_weight, "target_loss"))
            if control_weight > 0:
                loss_config.append((control_weight, "control_loss"))
            if attention_weight > 0:
                loss_config.append((attention_weight, "attention_loss"))
        else:
            loss_config = None
        main_device = self.models[0].device
        control_candses = []
        losses = []
        modes = ["control"]
        
        curr_control_dict = {
            'control': self.control_str
        }
        for mode in modes:
            control_cands = []
            
            curr_control = curr_control_dict[mode]
            grad = None
            for j, worker in enumerate(self.workers):
                new_grad = worker(self.prompts[j], "grad", worker.model, mode, target_weight, attention_pooling_method, attention_weight, attention_weight_dict).to(main_device)
                new_grad = new_grad / new_grad.norm(dim=-1, keepdim=True)
                if grad is None:
                    grad = torch.zeros_like(new_grad)
                if grad.shape != new_grad.shape:
                    with torch.no_grad():
                        control_cand = self.prompts[j-1].sample_control(grad, mode, batch_size, topk, temp, allow_non_ascii)
                        control_cands.append(self.get_filtered_cands(j-1, control_cand, filter_cand=filter_cand, curr_control=curr_control))
                    grad = new_grad
                else:
                    grad += new_grad
                with torch.no_grad():
                    control_cand = self.prompts[j].sample_control(grad, mode, batch_size, topk, temp, allow_non_ascii)
                    control_cands.append(self.get_filtered_cands(j, control_cand, filter_cand=filter_cand, curr_control=curr_control))
                    
            del grad, control_cand ; gc.collect()
            control_candses.append(control_cands)
            
            # Search
            loss = torch.zeros(len(control_cands) * batch_size).to(main_device)
            with torch.no_grad():
                for j, cand in enumerate(control_cands):
                    # Looping through the prompts at this level is less elegant, but
                    # we can manage VRAM better this way
                    progress = tqdm(range(len(self.prompts[0])), total=len(self.prompts[0])) if verbose else enumerate(self.prompts[0])
                    for i in progress:
                        if merge_loss_compute:
                            _, _, _, loss_infos = zip(*[worker(self.prompts[k][i],
                                                        "logits",
                                                        worker.model,
                                                        cand,
                                                        mode,
                                                        attention_pooling_method, 
                                                        attention_weight_dict,
                                                        return_ids=True,
                                                        enable_prefix_sharing=enable_prefix_sharing,
                                                        loss_config=loss_config) for k, worker in enumerate(self.workers)])
                            
                            for loss_info in loss_infos:
                                for name, (weight, loss_i) in loss_info.items():
                                    loss[j * batch_size:(j + 1) * batch_size] += weight * loss_i.to(main_device)
                                    print(f'{name}_min: {loss_i.argmin()}, {loss_i.min()}')
                               
                        else:
                            logits, attentions, ids = zip(*[worker(self.prompts[k][i],
                                                        "logits",
                                                        worker.model,
                                                        cand,
                                                        mode,
                                                        attention_pooling_method, 
                                                        attention_weight_dict,
                                                        return_ids=True,
                                                        enable_prefix_sharing=enable_prefix_sharing,
                                                        loss_config=loss_config) for k, worker in enumerate(self.workers)])
                            
                            attns = attentions[-1]
                            loss1 = sum([
                                target_weight*self.prompts[k][i].target_loss(logit, id).mean(dim=-1).to(main_device) 
                                for k, (logit, id) in enumerate(zip(logits, ids))
                            ])
                            loss[j*batch_size:(j+1)*batch_size] += loss1
                            print(f'loss_target:{loss1.argmin()}, {loss1.min()}')
                            loss2 = sum([
                                attention_weight*self.prompts[k][i].attention_loss(attention).to(main_device) 
                                for k, attention in enumerate(attns)
                            ])
                            loss[j*batch_size:(j+1)*batch_size] += loss2
                            print(f'loss_attention:{loss2.argmin()}, {loss2.min()}')
                            
                            del logits, ids , attentions, loss1, loss2 ; gc.collect()

                        if verbose:
                            progress.set_description(f"loss_{mode}={loss[j*batch_size:(j+1)*batch_size].min().item()/(i+1):.4f}")
                    
                    index = loss[j * batch_size:(j + 1) * batch_size].argmin()
                    print(f'choose: {index}')
                        
            losses.append(loss)
            
        min_loss = torch.tensor([loss.min() for loss in losses])
        if modes[min_loss.argmin()] == "control":
            min_idx = losses[0].argmin()
            model_idx = min_idx // batch_size
            batch_idx = min_idx % batch_size
            next_control, cand_loss = control_candses[0][model_idx][batch_idx], losses[0][min_idx]
            print('Current control length:', len(self.workers[0].tokenizer(next_control, add_special_tokens=False).input_ids))
            print(f"next_control: {next_control}")
            
        del control_candses, losses ; gc.collect()
        
        return next_control, cand_loss.item() / len(self.prompts[0]) / len(self.workers)
