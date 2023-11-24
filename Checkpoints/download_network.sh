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
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=13HBM1t1rnks0M8N7xvC8NK8VP9iMTasq' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=13HBM1t1rnks0M8N7xvC8NK8VP9iMTasq" -O QSMnet_64.tar.gz && rm -rf /tmp/cookies.txt  

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1YeQ_kNpB1W5vok96sTNzNE7_uP8pCFyU' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1YeQ_kNpB1W5vok96sTNzNE7_uP8pCFyU" -O QSMnet+_64.tar.gz && rm -rf /tmp/cookies.txt 

tar -xvzf QSMnet_64.tar.gz

tar -xvzf QSMnet+_64.tar.gz

rm QSMnet_64.tar.gz

rm QSMnet+_64.tar.gz
