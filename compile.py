import os,subprocess,sys

goals = ["arch","base64","basename","cat","chcon","chgrp","chmod",
"chown","chroot","cksum","comm","cp","csplit","cut","date","dd",
"df","dir","dircolors","dirname","du","echo","env","expand","expr",
"factor","false","fmt","fold","groups","head","hostid","hostname",
"id","install","join","kill","link","ln","logname","ls","md5sum",
"mkdir","mkfifo","mknod","mktemp","mv","nice","nl","nohup",
"nproc","numfmt","od","paste","pathchk","pinky","pr","printenv",
"printf","ptx","pwd","readlink","realpath","rm","rmdir","runcon",
"seq","shred","shuf","sleep","sort","split","stat","stdbuf","stty",
"sum","tac","tail","tee","test","timeout","touch","tr","true",
"truncate","tsort","tty","uname","unexpand","uniq","unlink",
"uptime","users","vdir","wc","who","whoami","yes"]

opts = "optimization_flags.txt"
subprocess.call(["mkdir","binaries"])

with open(opts, "r") as f:
    flags = f.readlines()
os.chdir("coreutils")
subprocess.call(["./bootstrap"])
os.chdir("..")
for flag in flags:
    print("\n\n\n")
    print("COMPILING WITH FLAG: "+flag)
    print("\n\n\n")
    flag = flag.replace("\n","")
    args = flag + " -frecord-gcc-switches"
    print("\n\n"+args+"\n\n")
    subprocess.call(["mkdir","binaries/"+flag])
    try:
        os.chdir("coreutils")
        subprocess.call(["./configure","CFLAGS="+args])
        subprocess.call(["make"])
        for goal in goals:
            if goal in os.listdir("src"):
                subprocess.call(["mv","src/"+goal,"../binaries/"+flag+"/"+goal])
        subprocess.call(["make","clean"])
        os.chdir("..")
    except Error2:
        print("ERROR AT "+flag+"\n"+Error2)
        if "compile.py" in os.listdir(".."):
            os.chdir("..")
