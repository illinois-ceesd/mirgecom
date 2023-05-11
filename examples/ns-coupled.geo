Point(1) = {-0.5,-2.0,0.0};
Point(2) = { 0.5,-2.0,0.0};
Point(3) = { 0.5, 0.0,0.0};
Point(4) = {-0.5, 0.0,0.0};
Point(5) = { 0.5, 2.0,0.0};
Point(6) = {-0.5, 2.0,0.0};

Line(1) = {1,2};
Line(2) = {2,3};
Line(3) = {3,4};
Line(4) = {4,1};
Line(5) = {3,5};
Line(6) = {5,6};
Line(7) = {6,4};

Transfinite Line {1} = 11 Using Progression 1.0;
Transfinite Line {2} = 21 Using Progression 1.0;
Transfinite Line {3} = 11 Using Progression 1.0;
Transfinite Line {4} = 21 Using Progression 1.0;
Transfinite Line {5} = 21 Using Progression 1.0;
Transfinite Line {6} = 11 Using Progression 1.0;
Transfinite Line {7} = 21 Using Progression 1.0;

Line Loop(11) = { 1,2,3,4};
Line Loop(12) = {-3,5,6,7};

Plane Surface(11) = {-11};
Plane Surface(12) = {-12};

Physical Surface("Lower") = {11};
Physical Surface("Upper") = {12};

Physical Curve("Lower Bottom") = {1};
Physical Curve("Upper Top") = {6};
Physical Curve("Lower Sides") = {4,2};
Physical Curve("Upper Sides") = {7,5};
