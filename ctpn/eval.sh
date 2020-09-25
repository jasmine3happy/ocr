#! /bin/bash
python ctpn_predict.py
cd evaluation
python zip.py
python script.py -c DetEva
