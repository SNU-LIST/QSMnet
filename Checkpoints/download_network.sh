#
# Description :
#   Download code of QSMnet and QSMnet+ files.
#   Network files are located in Google drive link:
#   https://drive.google.com/drive/folders/1kSxj1Pw3yC_NC9jEkWZ8XxDdLuqpJeaf
#
# Copyright @ Woojin Jung & Jaeyeon Yoon
# Laboratory for Imaging Science and Technology
# Seoul National University
# email : dhcntjr9696@snu.ac.kr
#
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1kyRKSPN78i1npIaFvJJJgmB84v-N106m' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1kyRKSPN78i1npIaFvJJJgmB84v-N106m" -O QSMnet_64.tar.gz && rm -rf /tmp/cookies.txt  

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1ee5nI-MMySImX2tM4m71szcjFow4DD5g' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1ee5nI-MMySImX2tM4m71szcjFow4DD5g" -O QSMnet+_64.tar.gz && rm -rf /tmp/cookies.txt 

tar -xvzf QSMnet_64.tar.gz

tar -xvzf QSMnet+_64.tar.gz

rm QSMnet_64.tar.gz

rm QSMnet+_64.tar.gz
