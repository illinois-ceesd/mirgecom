Point( 1) = {-0.50,-0.05,0.0};
Point( 2) = {-0.50, 0.05,0.0};
Point( 3) = {-0.20,-0.05,0.0};
Point( 4) = {-0.20, 0.05,0.0};
Point( 5) = {-0.10,-0.05,0.0};
Point( 6) = {-0.10, 0.05,0.0};
Point( 7) = { 0.00,-0.05,0.0};
Point( 8) = { 0.00, 0.05,0.0};
Point( 9) = {+0.10,-0.05,0.0};
Point(10) = {+0.10, 0.05,0.0};
Point(11) = {+0.20,-0.05,0.0};
Point(12) = {+0.20, 0.05,0.0};
Point(13) = {+0.50,-0.05,0.0};
Point(14) = {+0.50, 0.05,0.0};

Line( 1) = {1,2};

Line( 2) = {2,4};
Line( 3) = {4,6};
Line( 4) = {6,8};
Line( 5) = {8,10};
Line( 6) = {10,12};
Line( 7) = {12,14};
Line( 8) = {14,13};
Line( 9) = {13,11};
Line(10) = {11,9};
Line(11) = {9,7};
Line(12) = {7,5};
Line(13) = {5,3};
Line(14) = {3,1};

Line(15) = {4,3};
Line(16) = {6,5};
Line(18) = {10,9};
Line(19) = {12,11};

Transfinite Line {1} =  6 Using Progression 1.0;
Transfinite Line {2} = 16 Using Progression 1.0;
Transfinite Line {3} =  6 Using Progression 1.0;
Transfinite Line {4} =  6 Using Progression 1.0;
Transfinite Line {5} =  6 Using Progression 1.0;
Transfinite Line {6} =  6 Using Progression 1.0;
Transfinite Line {7} = 16 Using Progression 1.0;
Transfinite Line {8} =  6 Using Progression 1.0;
Transfinite Line {9} = 16 Using Progression 1.0;
Transfinite Line {10} =  6 Using Progression 1.0;
Transfinite Line {11} =  6 Using Progression 1.0;
Transfinite Line {12} =  6 Using Progression 1.0;
Transfinite Line {13} =  6 Using Progression 1.0;
Transfinite Line {14} = 16 Using Progression 1.0;
Transfinite Line {15} =  6 Using Progression 1.0;
Transfinite Line {16} =  6 Using Progression 1.0;
Transfinite Line {17} =  6 Using Progression 1.0;
Transfinite Line {18} =  6 Using Progression 1.0;
Transfinite Line {19} =  6 Using Progression 1.0;

Line Loop(11) = { 1,2,15,14};
Line Loop(12) = {-15,3,16,13};
Line Loop(13) = {-16,4,5,18,11,12};
Line Loop(14) = {-18,6,19,10};
Line Loop(15) = {-19,7,8,9};

Plane Surface(11) = {11};
Plane Surface(12) = {12};
Plane Surface(13) = {13};
Plane Surface(14) = {14};
Plane Surface(15) = {15};

Physical Surface("Fluid") = {11,15};
Physical Surface("Sample") = {12,14};
Physical Surface("Holder") = {13};

Physical Curve("Fluid Hot") = {1};
Physical Curve("Fluid Cold") = {8};
Physical Curve("Fluid Sides") = {2,7,9,14};
Physical Curve("Sample Sides") = {3,6,10,13};
Physical Curve("Holder Sides") = {4,5,11,12};
