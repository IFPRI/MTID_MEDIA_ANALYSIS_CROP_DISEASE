source /home/gate/.bashrc
conda activate ifpriDisease
python /home/gate/repos/MTID_MEDIA_ANALYSIS_CROP_DISEASE/ifpriDiseaseApp.py --config /home/gate/repos/MTID_MEDIA_ANALYSIS_CROP_DISEASE//diseaseTypeTrain.config --gpu --rss --outputFolder /home/gate/repos/MTID_MEDIA_ANALYSIS_CROP_DISEASE/output/ --model /home/gate/repos/MTID_MEDIA_ANALYSIS_CROP_DISEASE/trainedModel/
sleep 10
pkill screen
