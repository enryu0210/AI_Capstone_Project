set -ex
python test.py --dataroot /content/dataset/dataset/DesmokeData/combined  \
               --name pfan_invivo \
               --model pix2pix \
               --netG pfan \
               --direction AtoB \
               --dataset_mode aligned \
               --norm batch \
               --phase test\
               --ndf 64 \
               --ngf 64  

               

