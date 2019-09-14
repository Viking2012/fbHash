def chunker(bytestream, n=7):
    for i in range(n, len(bytestream)+1):
        yield bytestream[i-n:i]


def fbhashB():
    with open('fbHash/test/test1.txt','rb') as f:
        raw = f.read()
        f1 = list(chunker(raw))
        rh1 = RollingHash(raw,7)
    with open('fbHash/test/test2.txt','rb') as f:
        raw = f.read()
        f2 = list(chunker(raw))

    print(len(f1), f1, rh1)
    print(len(f2), f2)

    return f1

class RollingHash:
	def __init__(self, string, size):
		self.str  = string
		self.hash = 0
		
		for i in range(0, size):
			self.hash += ord(self.str[i])
		
		self.init = 0
		self.end  = size
		
	def update(self):
		if self.end <= len(self.str) -1:
			self.hash -= ord(self.str[self.init])
			self.hash += ord(self.str[self.end])
			self.init += 1
			self.end  += 1
			
	def digest(self):
		return self.hash

	def text(self):
		return self.str[self.init:self.end]


if __name__ == "__main__":
    fbhashB()