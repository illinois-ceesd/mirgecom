//

lcar0 = 0.002;
lcar1 = 0.020;

xI = 0.04;
xO = 0.04;
yy = 0.04;
Point(1)  = { -xI, 0.0, 0.0, lcar1};
Point(2)  = { 0.0, 0.0, 0.0, lcar0};
Point(3)  = { +xO, 0.0, 0.0, lcar1};
Point(4)  = { +xO, +yy, 0.0, lcar1};
Point(5)  = { 0.0, +yy, 0.0, lcar0};
Point(6)  = { -xI, +yy, 0.0, lcar1};

//Define bounding box edges
Line( 1) = {1, 2};
Line( 2) = {2, 3};
Line( 3) = {3, 4};
Line( 4) = {4, 5};
Line( 5) = {5, 6};
Line( 6) = {6, 1};
Line( 7) = {2, 5};

Transfinite Line {-1} = 27 Using Progression 1.1;
Transfinite Line { 5} = 27 Using Progression 1.1;

Transfinite Line {3} = 61 Using Progression 1.05;
Transfinite Line {-6} = 61 Using Progression 1.05;
Transfinite Line {7} = 61 Using Progression 1.05;

Transfinite Line {2} = 61 Using Progression 1.02;
Transfinite Line {-4} = 61 Using Progression 1.02;

Line Loop(1) = {6,1,7,5};
Plane Surface(1) = {-1};
Transfinite Surface {1} Alternate;

Line Loop(2) = {4,-7,2,3};
Plane Surface(2) = {-2};
Transfinite Surface {2} Alternate;

Physical Line("inlet") = {6};
Physical Line("outlet") = {3};
Physical Line("slip") = {1};
Physical Line("wall") = {2};
Physical Line("farfield") = {4,5};
Physical Surface("domain") = {1,2};

