Point(1) = {-0.2,-0.05,0.0};
Point(2) = {-0.2, 0.05,0.0};
Point(3) = { 0.0, 0.05,0.0};
Point(4) = { 0.0,-0.05,0.0};
Point(5) = {+0.1, 0.05,0.0};
Point(6) = {+0.1,-0.05,0.0};

Line(1) = {1,2};
Line(2) = {2,3};
Line(3) = {3,4};
Line(4) = {4,1};
Line(5) = {3,5};
Line(6) = {5,6};
Line(7) = {6,4};

Transfinite Line {1} =  6 Using Progression 1.0;
Transfinite Line {2} = 11 Using Progression 1.0;
Transfinite Line {3} =  9 Using Progression 1.0;
Transfinite Line {4} = 11 Using Progression 1.0;
Transfinite Line {5} =  6 Using Progression 1.0;
Transfinite Line {6} =  6 Using Progression 1.0;
Transfinite Line {7} =  6 Using Progression 1.0;

Line Loop(11) = { 1,2,3,4};
Line Loop(12) = {-3,5,6,7};

Plane Surface(11) = {11};
Plane Surface(12) = {12};

Physical Surface("Solid") = {11};
Physical Surface("Fluid") = {12};

Physical Curve("Solid Presc") = {1};
Physical Curve("Fluid Presc") = {6};
Physical Curve("Solid Sides") = {4,2};
Physical Curve("Fluid Sides") = {7,5};
