python3 Low\ var.py 

🔎 Checking unique value counts in numeric columns:

Destination Port: 23950 unique value(s)
Flow Duration: 187752 unique value(s)
Total Fwd Packets: 297 unique value(s)
Total Backward Packets: 367 unique value(s)
Total Length of Fwd Packets: 3831 unique value(s)
Total Length of Bwd Packets: 6760 unique value(s)
Fwd Packet Length Max: 1891 unique value(s)
Fwd Packet Length Min: 151 unique value(s)
Fwd Packet Length Mean: 7401 unique value(s)
Fwd Packet Length Std: 9555 unique value(s)
Bwd Packet Length Max: 1945 unique value(s)
Bwd Packet Length Min: 343 unique value(s)
Bwd Packet Length Mean: 8655 unique value(s)
Bwd Packet Length Std: 9650 unique value(s)
Flow Bytes/s: 202294 unique value(s)
Flow Packets/s: 194094 unique value(s)
Flow IAT Mean: 193666 unique value(s)
Flow IAT Std: 159622 unique value(s)
Flow IAT Max: 139745 unique value(s)
Flow IAT Min: 11093 unique value(s)
Fwd IAT Total: 78049 unique value(s)
Fwd IAT Mean: 95095 unique value(s)
Fwd IAT Std: 91184 unique value(s)
Fwd IAT Max: 77978 unique value(s)
Fwd IAT Min: 8820 unique value(s)
Bwd IAT Total: 101942 unique value(s)
Bwd IAT Mean: 117311 unique value(s)
Bwd IAT Std: 113859 unique value(s)
Bwd IAT Max: 101379 unique value(s)
Bwd IAT Min: 3787 unique value(s)
Fwd PSH Flags: 2 unique value(s)
Fwd Header Length: 714 unique value(s)
Bwd Header Length: 806 unique value(s)
Fwd Packets/s: 192230 unique value(s)
Bwd Packets/s: 144444 unique value(s)
Min Packet Length: 109 unique value(s)
Max Packet Length: 2352 unique value(s)
Packet Length Mean: 11680 unique value(s)
Packet Length Std: 13502 unique value(s)
Packet Length Variance: 13502 unique value(s)
FIN Flag Count: 2 unique value(s)
SYN Flag Count: 2 unique value(s)
RST Flag Count: 2 unique value(s)
PSH Flag Count: 2 unique value(s)
ACK Flag Count: 2 unique value(s)
URG Flag Count: 2 unique value(s)
ECE Flag Count: 2 unique value(s)
Down/Up Ratio: 8 unique value(s)
Average Packet Size: 11270 unique value(s)
Avg Fwd Segment Size: 7401 unique value(s)
Avg Bwd Segment Size: 8655 unique value(s)
Fwd Header Length.1: 714 unique value(s)
Subflow Fwd Packets: 297 unique value(s)
Subflow Fwd Bytes: 3831 unique value(s)
Subflow Bwd Packets: 367 unique value(s)
Subflow Bwd Bytes: 6760 unique value(s)
Init_Win_bytes_forward: 1804 unique value(s)
Init_Win_bytes_backward: 1922 unique value(s)
act_data_pkt_fwd: 234 unique value(s)
min_seg_size_forward: 8 unique value(s)
Active Mean: 40409 unique value(s)
Active Std: 5669 unique value(s)
Active Max: 40313 unique value(s)
Active Min: 38393 unique value(s)
Idle Mean: 35285 unique value(s)
Idle Std: 5857 unique value(s)
Idle Max: 33002 unique value(s)
Idle Min: 48018 unique value(s)
Label: 2 unique value(s)
Flow Bytes/s_IS_MISSING: 2 unique value(s)

🧹 Result:
✅ All numeric columns have more than 1 unique value.
(myenv) ┌─[kay@parrot]─[~/Documents/Workspace-S25/SE/SeProject/APT DETECTION/z.after mids progress/processing/New Approach by HH]
└──╼ $python3 low-var.py 

🧹 Checking for constant or imbalanced binary features...

⚠️ FIN Flag Count has 2 values but is imbalanced (99.73% vs 0.27%).
⚠️ RST Flag Count has 2 values but is imbalanced (99.99% vs 0.01%).
⚠️ ECE Flag Count has 2 values but is imbalanced (99.99% vs 0.01%).
⚠️ Flow Bytes/s_IS_MISSING has 2 values but is imbalanced (100.00% vs 0.00%).

📉 Dropping low variance/imbalanced columns: ['FIN Flag Count', 'RST Flag Count', 'ECE Flag Count', 'Flow Bytes/s_IS_MISSING']

📁 Final cleaned dataset saved to: dataset_after_low_variance_removed.csv
(myenv) ┌─[kay@parrot]─[~/Documents/Workspace-S25/SE/SeProject/APT DETECTION/z.after mids progress/processing/New Approach by HH]
└──╼ $