//

lcar0 = 0.010;
lcar1 = 0.050;


Point(1)  = { +0.0, 0.0, 0.0, lcar1};
Point(2)  = { +0.6, 0.0, 0.0, lcar0};
Point(3)  = { +0.6, 0.2, 0.0, lcar0};
Point(4)  = {  3.0, 0.2, 0.0, lcar1};
Point(5)  = {  3.0, 1.0, 0.0, lcar1};
Point(6)  = {  0.0, 1.0, 0.0, lcar1};

//Define bounding box edges
Line( 1) = {1, 2};
Line( 2) = {2, 3};
Line( 3) = {3, 4};
Line( 4) = {4, 5};
Line( 5) = {5, 6};
Line( 6) = {6, 1};



Line Loop(1) = {1:6};
Plane Surface(1) = {-1};
/*Transfinite Surface {1};*/

/*dx = 0.000025;*/
/*dy = 0.000125;*/
/*Point(21)  = {  0.007 + dx, 0.00, 0.0, lcar0};*/
/*Point(22)  = {  0.007 - dx, 0.00, 0.0, lcar0};*/
/*Point(23)  = {  0.007     ,  +dy, 0.0, lcar0};*/
/*Point(24)  = {  0.007     ,  -dy, 0.0, lcar0};*/

Physical Line("inlet") = {6};
Physical Line("outlet") = {4};
Physical Line("wall") = {1,2,3,5};
Physical Surface("domain") = {1};

