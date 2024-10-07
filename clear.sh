cd data/advbench
rm  `ls | grep "blank"`
rm  `ls | grep "transfer"`
rm  `ls | grep "direct"`

cd ../../experiments
rm  -rf `ls | grep "results"`
rm -rf testcase

cd bash_scripts
rm -rf `ls | grep "out_"`

cd ../../eval
rm -rf generations
rm -rf testcases

cd transfer_across_goals
rm -rf raw_testcase
rm -rf use_testcase
rm  `ls | grep "blank"`

cd