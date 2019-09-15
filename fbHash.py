import pprint
from random import randrange

def chunker(bytestream, n=7):
    for i in range(n, len(bytestream)+1):
        yield bytestream[i-n:i]


def fbhashB():
    f1 = []
    f2 = []
    with open('fbHash/data/test1.txt','rb') as f:
        raw = f.read()
        rh = RollingHash(raw,7)
        for i,_ in enumerate(range(7, len(raw) + 1),start=1):
            f1.append({
                'i':i,
                'hash':rh.digest(),
                'text':rh.text(),
            })
            rh.update()
    with open('fbHash/data/test2.txt','rb') as f:
        raw = f.read()
        rh = RollingHash(raw,7)
        for i,_ in enumerate(range(7, len(raw) + 1),start=1):
            f2.append({
                'i':i,
                'hash':rh.digest(),
                'text':rh.text(),
            })
            rh.update()

    mismatched = 0
    all_hashes = set()

    for yin, yang in zip(f1,f2):
        if yin['hash'] != yang['hash']:
            all_hashes.add(yin['hash'])
            all_hashes.add(yang['hash'])
            print(yin['text'], yang['text'])
            mismatched += 1

    percent_matched = 1 - (mismatched/len(f1))
    print(f"f1 and f2 were {percent_matched:.2%}% similar")
    print(sorted(all_hashes))


class RollingHash:
    known_64_bit_primes = [
            17586613600806056593,
            10324706610870574883,
            14385965969526276271,
            15700719402893486197,
            13390804203280917121,
            12631952504492069741,
            14687623246052906689,
            18235099962527857067,
            13557970565612484931,
        ]
    def __init__(self, bytestring, k=7, a=26, mod=None):
        self.b  = bytestring
        self.hash = 0
        self.a = a

        if mod is None:
            self.mod = self.large_primes(1)
        else:
            self.mod = mod
		
        for i in range(0, k):
            self.hash = self._finalize_hash(self.hash, 0, self.b[i])

        self.init = 0
        self.end  = k
		
    def update(self):
        if self.end <= len(self.b) - 1:
            old = self.b[self.init]
            new = self.b[self.end]
            self.hash = self._finalize_hash(self.hash, old, new)
            self.init += 1
            self.end  += 1

    def _finalize_hash(self, current, old, new):
        current -= old * self.a
        current += new * self.a
        # current -= old
        # current += new
        current += self.mod
        return current % self.mod
            
    def digest(self):
        return self.hash

    def text(self,encoding='utf-8'):
        return self.b[self.init:self.end].decode(encoding)

    def large_primes(self,idx=None):
        if idx is None:
            idx = randrange(len(self.known_64_bit_primes))
        return self.known_64_bit_primes[idx]


if __name__ == "__main__":
    # print(large_primes())
    fbhashB()