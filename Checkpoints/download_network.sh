#
# Description :
#   Download code of QSMnet and QSMnet+ files.
#   Network files are located in Google drive link:
#   https://drive.google.com/drive/folders/1fQKauCatiJezHfbre9yjqYgM1LwtwJpv
#
# Copyright @ Woojin Jung & Jaeyeon Yoon
# Laboratory for Imaging Science and Technology
# Seoul National University
# email : dhcntjr9696@snu.ac.kr
#
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1C4aV6FHXhZ_aasDQFxVVtISKQNN-mA_n' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1C4aV6FHXhZ_aasDQFxVVtISKQNN-mA_n" -O QSMnet_64.tar.gz && rm -rf /tmp/cookies.txt  

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1fFYW-Az6lBgyo10Km5CryLAorRrX5dou' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1fFYW-Az6lBgyo10Km5CryLAorRrX5dou" -O QSMnet+_64.tar.gz && rm -rf /tmp/cookies.txt 

tar -xvzf QSMnet_64.tar.gz

tar -xvzf QSMnet+_64.tar.gz

rm QSMnet_64.tar.gz

rm QSMnet+_64.tar.gz
