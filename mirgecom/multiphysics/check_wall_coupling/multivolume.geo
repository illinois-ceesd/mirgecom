Point(1) = {-20,-20,0.0};
Point(2) = {20,-20,0.0};
Point(3) = {20,0.0,0.0};
Point(4) = {-20,0.0,0.0};
Point(5) = {20,20,0.0};
Point(6) = {-20,20,0.0};

Line(1) = {1,2};
Line(2) = {2,3};
Line(3) = {3,4};
Line(4) = {4,1};
Line(5) = {3,5};
Line(6) = {5,6};
Line(7) = {6,4};

Line Loop(1) = {1,2,3,4};
Line Loop(2) = {-3,5,6,7};

// Not sure why these need to be flipped
Plane Surface(1) = {-1};
Plane Surface(2) = {-2};

Transfinite Line {1} = 41 Using Progression 1.0;
Transfinite Line {2} = 21 Using Progression 1.0;
Transfinite Line {3} = 41 Using Progression 1.0;
Transfinite Line {4} = 21 Using Progression 1.0;
Transfinite Line {5} = 21 Using Progression 1.0;
Transfinite Line {6} = 41 Using Progression 1.0;
Transfinite Line {7} = 21 Using Progression 1.0;

Physical Surface("Lower") = {1};
Physical Surface("Upper") = {2};

Physical Curve("Lower Sides") = {1,2,4};
Physical Curve("Upper Sides") = {5,6,7};
