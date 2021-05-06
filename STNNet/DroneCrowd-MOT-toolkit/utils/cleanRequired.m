function cl=cleanRequired(seqFolder)

cl = ~isempty(strfind(seqFolder,'MOT16')) || ~isempty(strfind(seqFolder,'MOT17'));