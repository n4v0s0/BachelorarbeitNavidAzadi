from nltk import ngrams
import sys,subprocess,collections,os,time
inst = []
if sys.argv[1] == "":
    print("Bitte angeben, mit welchem N die N-Gramme berechnet werden sollen.")
    sys.exit()
N = int(sys.argv[1])

#Ziel Ordner für N-Gram Dateien erstellen
subprocess.Popen(["mkdir","ngramOutput"])
#Assembly Instruktionen einlesen
with open("verwendeterCode/asm_instructions.txt") as f:
	lines = f.readlines()
	for line in lines:
		line = line.replace("\n","")
		inst.append(line.lower())
#Durch alle Binaries durch iterieren
for dirc in os.listdir("binaries"):
    path = os.path.join("binaries",dirc)
    #dabei befindet sich in jedem Ordner eine Menge von Binaries die mit einer bestimmten Optimierung erstellt worden sind
    subprocess.Popen(["mkdir","ngramOutput/"+dirc])
    for binary in os.listdir(path):
        #N-Gramme mithilfe von Objdump und der ngram-Funktion von NLTK erstellen
        filename = os.path.join(path,binary)
        dump = subprocess.Popen(["objdump","-D",filename],stdout=subprocess.PIPE)
        dump = dump.stdout.read()
        check = False
        opcode = []
        for i in dump.split():
                if b'.text:' == i:
                        check = True
                if check == True:
                        if i.decode('utf-8') in inst:
                                opcode.append(i.decode('utf-8'))
                if b'Disassembly' == i:
                        check = False
        grams = ngrams(opcode,N)
        result = collections.Counter(grams)
        result = str(result).split(',')
        i = 0
        j = 0
        final_res = []
        curr = ""
        for line in result:
                i += 1
                j += 1
                curr += line
                if i > N-1 and i % N == 0:
                        final_res.append(curr)
                        curr = ""
                        i = 0
        final_res[0] = final_res[0][9:]
        #Ergebnis-Ordner für jeweilige Optimierung erstellen und N-Gramme in Datei schreiben
        subprocess.Popen(["touch","ngramOutput/"+dirc+"/"+binary+"_NGRAMS.txt"])
        with open("ngramOutput/"+dirc+"/"+binary+"_NGRAMS.txt", "w") as f:
                for line in final_res:
                        f.write(line+"\n")




