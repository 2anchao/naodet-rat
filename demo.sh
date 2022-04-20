cd /project/train/src_repo/naodet-rat &&\
python setup.py develop &&\
python demo/demo.py image --config config/nanodet_custom_lulu_xml_dataset.yml --model /project/train/models/model_best/nanodet_model_best.pth --path /home/data/888/176.jpg --save_result