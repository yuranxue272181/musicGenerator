# musicGenerator

#py
1. generate_music.py        使用已经训练好的generator来生成midi
2. generate_music_music.py  使用已经训练好的generator来生成只含piano的midi
3. midi_utils.py            使用music21将database中的midi筛选并且fix
4. train_gan.py             第一版GAN，D过于自信导致G始终为100无法学习
5. train_gan_version2       第二版GAN，使得G的训练次数翻倍，添加小型奖励机制
                             20 epochs -> D=0.3238, G=4.1978,生成结果杂乱无序，奖励机制并不完善，无规律性，D仍旧过于自信
