#coding: utf-8

#
# Find the mapping of click_id in test_supplement.csv and test.csv
#

test_file = "../data/test.csv"
test_s_file = "../data/test_supplement.csv"
sig_map = {}
total = 0
with open(test_file) as f:
	idx = 0
	for line in f:
		if idx ==0:
			idx += 1
			continue
		
		idx += 1
		flds = line.strip().split(",")
		click_id = int(flds[0])
		sig = ','.join(flds[1:])
		ret = sig_map.get(sig,[])
		total += 1
		if len(ret) <=0:
			sig_map[sig]=[click_id]
		else:
			sig_map[sig].append(click_id)	

print("Total len={},uniq_len={}".format(total, len(sig_map)))

out_fd =  open("supplement2test.txt","w") 
with open(test_s_file) as f:
	idx = 0
	for line in f:
		if idx == 0:
			idx += 1
			continue
		idx += 1
		flds = line.strip().split(",")
		click_id = int(flds[0])
		sig = ','.join(flds[1:])
		ret = sig_map.get(sig,None)
		if ret is not None:
			for click_id1 in ret:
					out_fd.write("{}\t{}\n".format(click_id,click_id1))
			del(sig_map[sig])

out_fd.close()
