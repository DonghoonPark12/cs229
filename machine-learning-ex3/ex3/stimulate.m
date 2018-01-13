x = linspace(-100, 100);
y = linspace(-100, 100);



[X,Y]=meshgrid(x,y);

z_plot = fun(X,Y);
%surf(z_plot);
%plot3(x,y,z_plot);
axis([-200 200 -200 200]);
mesh(x,y,z_plot);
%contour(x,y,z_plot);

%surf(x,y,z_plot);