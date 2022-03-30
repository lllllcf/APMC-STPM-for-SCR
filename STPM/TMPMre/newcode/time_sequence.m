  res = [0.1666, 1 ,  1/3, 1.5-0.1666-1-1/3,0;
        0,0.0833, 1/3, 1,  1.5-0.0833-1-1/3;
        0,0.3337, 1/3, 1.5-0.3337-1/3, 0;
        0,0.9166, 1/3, 1.5-1/3-0.9166,0;
         0.4145, 1 , 1.5-1-0.4145, 0,0;
         0,0.1688, 1/3,1.5-1/3-0.1688, 0;
         0,0.4188, 1/3, 1.5-1/3-0.4188, 0;
         0,0.9979, 1/3, 1.5-1/3-0.9979, 0;
           0,0.043,1/3,1,1.5-1-1/3-0.043;
           0,0.2543, 1/3, 1.5-1/3-0.2543, 0;
           0,0.5043, 1/3, 1.5-1/3-0.5043, 0;
           0.0877,1,1/3,1.5-1/3-1-0.0877,0];
 
       figure(1)
hold on
box on
set(gca,'linewidth',1.5,'FontSize',13,'FontWeight','bold')
%grid on
       
 b = barh(res,'stack');
 
 ylabel('\textbf{Element}','Interpreter','Latex','FontSize',15);
xlabel('\textbf{Normalized Time (s)}','Interpreter','Latex','FontSize',15);
legend({'$\phi = 0^{\circ}$','$\phi = 90^{\circ}$',...
    '$\phi = 180^{\circ}$','$\phi = 270^{\circ}$','OFF'},'FontSize',12,'Interpreter','Latex')

 phi = 0;
 theta = 20;
 T = 4;
 m=3;
 t1 = sind(theta)/T*(m-1)-1/6
 t2 = t1 + 1/4
 t3 = t1 + 1/2
 t4 = t1 - 1/4
 tau = 1/3;
