#
# Description :
#   Download code of QSMnet and QSMnet+ files.
#   Network files are located in Google drive link:
#   https://drive.google.com/drive/u/0/folders/1E7e9thvF5Zu68Sr9Mg3DBi-o4UdhWj-8 
#
# Copyright @ Woojin Jung & Jaeyeon Yoon
# Laboratory for Imaging Science and Technology
# Seoul National University
# email : wjjung93@snu.ac.kr
#
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1mxayxvumshJAHNhVnD8UOem4uD8C85TY' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1mxayxvumshJAHNhVnD8UOem4uD8C85TY" -O QSMnet_64.tar.gz && rm -rf /tmp/cookies.txt  
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1mxxxNlcxNEQVDModRtMKl5P8CeupSkMY' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1mxxxNlcxNEQVDModRtMKl5P8CeupSkMY" -O QSMnet+_64.tar.gz && rm -rf /tmp/cookies.txt 
tar -xvzf QSMnet_64.tar.gz
tar -xvzf QSMnet+_64.tar.gz
rm QSMnet_64.tar.gz
rm QSMnet+_64.tar.gz
