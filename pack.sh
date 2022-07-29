
dir="/home/notebook/code/group/intention_rec/TimeSeries/kdd2022/submissions" # dir of submission file, change accordingly

dest_dir=$(echo $dir"/src_ensemble")
mkdir $dest_dir
rm -rf $dest_dir/*

cp -r ./* $dest_dir

cd $dir
rm -rf $(find -type d -name .ipynb_checkpoints)
rm -rf $(find -type d -name __pycache__)
zip -r src_ensemble.zip src_ensemble
