import os
import argparse
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument("--in_path", type=str)
parser.add_argument("--out_path", type=str)
args = parser.parse_args()

def main():
  users = defaultdict(list)
  with open(args.in_path, 'r') as inf, open(args.out_path, 'w') as outf:
    for line in inf:
      uid, iid = line.strip().split(' ')
      # uid, iid = int(uid), int(iid)
      users[uid].append(iid)
    for uid, iids in users.items():
      outf.write("{}:{}\n".format(uid, ','.join(iids)))


if __name__ == "__main__":
  main()
