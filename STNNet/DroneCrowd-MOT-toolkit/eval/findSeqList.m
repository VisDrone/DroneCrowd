function nameSeqs = findSeqList(seqPath)

d = dir(seqPath);
nameSeqs = {d.name}';
nameSeqs(ismember(nameSeqs,{'.','..'})) = [];