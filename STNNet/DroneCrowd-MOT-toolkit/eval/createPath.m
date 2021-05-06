function createPath(newPath)

if(~isdir(newPath))
    mkdir(newPath);    
end