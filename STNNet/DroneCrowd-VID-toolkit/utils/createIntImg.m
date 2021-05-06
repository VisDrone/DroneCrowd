function intImg = createIntImg(img)

[height, width] = size(img);
intImg = img;
for i = 2:height
    intImg(i, 1) = intImg(i, 1) + intImg(i-1, 1);
end
for j = 2:width
    intImg(1, j) = intImg(1, j) + intImg(1, j-1);
end

for i = 2:height
    for j = 2:width
        intImg(i, j) = intImg(i, j) + intImg(i-1, j) + intImg(i, j-1) - intImg(i-1, j-1);
    end
end