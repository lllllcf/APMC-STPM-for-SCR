a = double(imread('formatlab.png'));
a1 = a(:,:,1);
a2 = a(:,:,2);
a3 = a(:,:,3);

[m,n] = size(a1);
d = 100;
for i = 1:m
    for j = 1:n
        r = a1(i,j);
        g = a2(i,j);
        b = a3(i,j);
        
        if sqrt((r-67)^2 + (g-195)^2 + (b-244)^2) <= d
            a1(i,j) = 33;
            a2(i,j) = 86;
            a3(i,j) = 232;
        end
    end
end

% a1(a1==67)=0;
% a2(a2==195)=0;
% a3(a3==244)=255;

a = cat(3,a1,a2,a3);

imshow(uint8(a));