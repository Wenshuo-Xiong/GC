% example: detail enhancement
% figure 6 in our paper

close all;

I = double(imread('tulips.bmp')) / 255;
p = I;

r = 16;
eps = 0.1^2;

q = zeros(size(I));

q(:, :, 1) = guidedfilter(I(:, :, 1), p(:, :, 1), r, eps);
q(:, :, 2) = guidedfilter(I(:, :, 2), p(:, :, 2), r, eps);
q(:, :, 3) = guidedfilter(I(:, :, 3), p(:, :, 3), r, eps);

I_enhanced = (I - q) * 5 + q;		%边缘加强

figure();
imshow([I,   q   ,  I_enhanced], [0, 1]);
	% 原图 滤波后   加强后
