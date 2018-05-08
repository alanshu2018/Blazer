#coding: utf-8

#
# Find the mapping of click_id in test_supplement.csv and test.csv
#

test_file = "../data/test.csv"
test_s_file = "../data/test_supplement.csv"


mapped ={}

fd = open(test_file)
sfd = open(test_s_file)
#skip header
fd.readline()
sfd.readline()

class FileLine(object):
    def __init__(self,fd, name):
        self.fd = fd
        self.name =name
        self.cnt = 0
        self.cur = None

    def _get_line(self):
        while True:
				line = self.fd.readline().strip()
				self.cnt +=1
				if self.cnt % 1000000 == 0:
					print("Processed {}: name={}".format(self.cnt,self.name))
				if len(line) <= 0:
					print("Invalid line {}: name={}".format(line,self.name))
					return None,None

				flds = line.strip().split(",")
				#time + other field
				return int(flds[0]),[flds[-1]] + flds[1:-1]

    def step_one(self):
        click_id, sig = self._get_line()
        self.cur = (click_id,sig)

    def get(self):
        if self.cur is None:
            self.step_one()
        return self.cur

fd_lines = FileLine(fd,"test")
sfd_lines = FileLine(sfd,"supplement")

def compute_mapping(fd, sfd):

    while True:
        id1,sig1 = fd.get()
        id2,sig2 = sfd.get()
        if id1 is None or id2 is None:
            break

        if sig1 == sig2:
            cnt = len(mapped)
            mapped[id2] = id1
            if cnt % 100000 == 0:
                print("sig1={}".format(sig1))
                print("sig2={}".format(sig2))
                print("Found {}->{}".format(id2,id1))

            fd.step_one()
            sfd.step_one()
        elif sig1 < sig2:
            fd.step_one()
        else:
            sfd.step_one()


compute_mapping(fd_lines, sfd_lines)

fd.close()
sfd.close()

with open("supplement2test.txt","w") as f:
    for k, v in mapped.items():
        f.write("{}\t{}\n".format(k,v))

