 mysql -h sun-feature-db.cvxxbx1dlxvk.us-west-2.rds.amazonaws.com -P 3306 -u ufcoroot -p sun_feature -t -e 'select * from DB_hmi_M_720s_nrt_wavelet_haar' > test.txt 
 mysql -h sun-feature-db.cvxxbx1dlxvk.us-west-2.rds.amazonaws.com -P 3306 -u ufcoroot -p sun_feature   --auto-vertical-output -e 'select * from DB_hmi_M_720s_nrt_wavelet_haar' > test-v.txt 

