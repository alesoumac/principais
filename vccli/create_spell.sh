cd ~/desenv/python/vcdoc
python create_spell.py modulo_ocr.dic train_imgs/test_imgs/*.json
rm modulo_ocr.dic
rm modulo_ocr.dic.new
rm train_imgs/test_imgs/*.new
rm train_imgs/validos/*.new
mv custom_spell.txt modulo_ocr.dic
cp modulo_ocr.dic ~/desenv/darknet/python/darknet_spell.dic
cp modulo_ocr.dic ~/desenv/gitprojs/dkn2/darknet_server/darknet_spell.dic
cp modulo_ocr.dic ~/desenv/gitprojs/dkn2/11291-plia/aisvb/vcdoc/inferencia/mvp/application/darknet_server/darknet_spell.dic
