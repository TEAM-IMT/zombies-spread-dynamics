clc; clear; %close all;

img_eleva = imread('Elevation_bw.bmp');
img_pop = imread('population-density-map.bmp');

movingp=...
[966, 3540; 2114, 1616; 2532, 3645; 3729,3021;
877, 2188; 145, 3466; 948,2992; 1308, 3123; 
1255, 1300; 416, 3700; 346, 2841; 3054, 4187;
4403, 3070; 3994, 3977; 1394, 2254; 2790, 1518;
1809, 3916; 4033, 2673; 2064, 3057; 3208, 1361];

fixedp=...
[1039, 2365; 1735, 353; 2580, 2247; 3727, 1718;
584, 1134; 236, 2555; 893, 1834; 1269, 1895; 
687, 267; 554, 2686; 252, 1904; 3157, 2810;
4451, 1927; 4064, 2708; 1138, 1048; 2478, 238;
1922, 2576; 4027, 1451; 2013, 1689; 2942, 132];

plot_points(img_eleva,movingp);
plot_points(img_pop,fixedp);
[C,tform] = reshape_image(img_eleva,movingp,fixedp);
[Brest, Rize] = find_cities(img_pop)
plot_overlap(img_pop,C);
imwrite(C,'elevation_fixed.bmp')

function [] = plot_points(img_eleva,x)
    n = 25;
    C = uint8(ones(size(img_eleva,1),size(img_eleva,2)));
    for i=1:length(x)
        C(x(i,2)-n:x(i,2)+n,x(i,1)-n:x(i,1)+n) = 255;
    end
    figure; imshow(img_eleva);
    hold on; h = imshow(C); hold off;
    set(h, 'AlphaData', 0.5); axis on;
end

function [C , tform] = reshape_image(img_eleva,x1,x2)
    tform = cp2tform(x1,x2,'polynomial');
    info = imfinfo('population-density-map.bmp');
    C = imtransform(img_eleva,tform,...
        'XData',[1 info.Width],'YData',[1 info.Height]);
end

function [Brest,Rize] = find_cities(C)
    for i=1:size(C,1)
        for j=1:size(C,2)
            if reshape(C(i,j,:),1,[])==uint8([255,0,0])
                Brest = [i,j];
            elseif reshape(C(i,j,:),1,[])==uint8([0,255,0])
                Rize = [i,j];
            end
        end
    end
end

function [] = plot_overlap(img_eleva,img_pop)
    figure;
    imshow(img_eleva); 
    hold on; 
    h = imshow(img_pop); 
    hold off;
    set(h, 'AlphaData', 0.5); 
    axis on;
end