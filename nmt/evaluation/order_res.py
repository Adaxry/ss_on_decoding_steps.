import sys
num = int(sys.argv[2])
with open(sys.argv[1], "r", encoding="utf-8") as in_f,\
  open(sys.argv[1] + ".ordered", "w", encoding="utf-8") as out_f:
  sent_ls = [""] * num
  for line in in_f:
    id_ = int(line.strip().split()[0])
    send = line.strip().split()[1:]
    assert id_ < num
    sent_ls[id_] = send
  for line in sent_ls:
    out_f.write(" ".join(line) + "\n")
