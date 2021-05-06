function nameSeqs = findSeqList(gtPath)

d = dir(gtPath);
nameSeqs = {d.name}';
nameSeqs(ismember(nameSeqs,{'.','..'})) = [];