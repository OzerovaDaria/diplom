îŔ!
ŃŁ
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
ž
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.3.02v2.3.0-rc2-23-gb36436b0878§
x
dense_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: @*
shared_namedense_8/kernel
q
"dense_8/kernel/Read/ReadVariableOpReadVariableOpdense_8/kernel*
_output_shapes

: @*
dtype0
p
dense_8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_8/bias
i
 dense_8/bias/Read/ReadVariableOpReadVariableOpdense_8/bias*
_output_shapes
:@*
dtype0
x
dense_9/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*
shared_namedense_9/kernel
q
"dense_9/kernel/Read/ReadVariableOpReadVariableOpdense_9/kernel*
_output_shapes

:@*
dtype0
p
dense_9/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_9/bias
i
 dense_9/bias/Read/ReadVariableOpReadVariableOpdense_9/bias*
_output_shapes
:*
dtype0
{
dense_10/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	0* 
shared_namedense_10/kernel
t
#dense_10/kernel/Read/ReadVariableOpReadVariableOpdense_10/kernel*
_output_shapes
:	0*
dtype0
s
dense_10/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_10/bias
l
!dense_10/bias/Read/ReadVariableOpReadVariableOpdense_10/bias*
_output_shapes	
:*
dtype0
{
dense_11/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@* 
shared_namedense_11/kernel
t
#dense_11/kernel/Read/ReadVariableOpReadVariableOpdense_11/kernel*
_output_shapes
:	@*
dtype0
r
dense_11/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_11/bias
k
!dense_11/bias/Read/ReadVariableOpReadVariableOpdense_11/bias*
_output_shapes
:@*
dtype0
z
dense_12/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@* 
shared_namedense_12/kernel
s
#dense_12/kernel/Read/ReadVariableOpReadVariableOpdense_12/kernel*
_output_shapes

:@*
dtype0
r
dense_12/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_12/bias
k
!dense_12/bias/Read/ReadVariableOpReadVariableOpdense_12/bias*
_output_shapes
:*
dtype0
{
dense_13/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@* 
shared_namedense_13/kernel
t
#dense_13/kernel/Read/ReadVariableOpReadVariableOpdense_13/kernel*
_output_shapes
:	@*
dtype0
s
dense_13/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_13/bias
l
!dense_13/bias/Read/ReadVariableOpReadVariableOpdense_13/bias*
_output_shapes	
:*
dtype0
{
dense_14/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@* 
shared_namedense_14/kernel
t
#dense_14/kernel/Read/ReadVariableOpReadVariableOpdense_14/kernel*
_output_shapes
:	@*
dtype0
r
dense_14/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_14/bias
k
!dense_14/bias/Read/ReadVariableOpReadVariableOpdense_14/bias*
_output_shapes
:@*
dtype0
z
dense_15/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@* 
shared_namedense_15/kernel
s
#dense_15/kernel/Read/ReadVariableOpReadVariableOpdense_15/kernel*
_output_shapes

:@*
dtype0
r
dense_15/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_15/bias
k
!dense_15/bias/Read/ReadVariableOpReadVariableOpdense_15/bias*
_output_shapes
:*
dtype0

NoOpNoOp
Ö6
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*6
value6B6 Bý5
˝
incoming_links
outcoming_links
create_message
link_update
readout
regularization_losses
trainable_variables
	variables
		keras_api


signatures
 
 
 
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
regularization_losses
trainable_variables
	variables
	keras_api
Ç
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
regularization_losses
trainable_variables
	variables
	keras_api
á
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
regularization_losses
trainable_variables
	variables
 	keras_api
 
v
!0
"1
#2
$3
%4
&5
'6
(7
)8
*9
+10
,11
-12
.13
/14
015
v
!0
"1
#2
$3
%4
&5
'6
(7
)8
*9
+10
,11
-12
.13
/14
015
­
1non_trainable_variables

2layers
regularization_losses
trainable_variables
	variables
3layer_metrics
4layer_regularization_losses
5metrics
 
|
6_inbound_nodes

!kernel
"bias
7regularization_losses
8trainable_variables
9	variables
:	keras_api
|
;_inbound_nodes

#kernel
$bias
<regularization_losses
=trainable_variables
>	variables
?	keras_api
 

!0
"1
#2
$3

!0
"1
#2
$3
­
@non_trainable_variables

Alayers
regularization_losses
trainable_variables
	variables
Blayer_metrics
Clayer_regularization_losses
Dmetrics
|
E_inbound_nodes

%kernel
&bias
Fregularization_losses
Gtrainable_variables
H	variables
I	keras_api
|
J_inbound_nodes

'kernel
(bias
Kregularization_losses
Ltrainable_variables
M	variables
N	keras_api
|
O_inbound_nodes

)kernel
*bias
Pregularization_losses
Qtrainable_variables
R	variables
S	keras_api
 
*
%0
&1
'2
(3
)4
*5
*
%0
&1
'2
(3
)4
*5
­
Tnon_trainable_variables

Ulayers
regularization_losses
trainable_variables
	variables
Vlayer_metrics
Wlayer_regularization_losses
Xmetrics
|
Y_inbound_nodes

+kernel
,bias
Zregularization_losses
[trainable_variables
\	variables
]	keras_api
f
^_inbound_nodes
_regularization_losses
`trainable_variables
a	variables
b	keras_api
|
c_inbound_nodes

-kernel
.bias
dregularization_losses
etrainable_variables
f	variables
g	keras_api
f
h_inbound_nodes
iregularization_losses
jtrainable_variables
k	variables
l	keras_api
|
m_inbound_nodes

/kernel
0bias
nregularization_losses
otrainable_variables
p	variables
q	keras_api
 
*
+0
,1
-2
.3
/4
05
*
+0
,1
-2
.3
/4
05
­
rnon_trainable_variables

slayers
regularization_losses
trainable_variables
	variables
tlayer_metrics
ulayer_regularization_losses
vmetrics
TR
VARIABLE_VALUEdense_8/kernel0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUEdense_8/bias0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEdense_9/kernel0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUEdense_9/bias0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEdense_10/kernel0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEdense_10/bias0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEdense_11/kernel0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEdense_11/bias0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEdense_12/kernel0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEdense_12/bias0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_13/kernel1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEdense_13/bias1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_14/kernel1trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEdense_14/bias1trainable_variables/13/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_15/kernel1trainable_variables/14/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEdense_15/bias1trainable_variables/15/.ATTRIBUTES/VARIABLE_VALUE
 

0
1
2
 
 
 
 
 

!0
"1

!0
"1
­
wnon_trainable_variables

xlayers
7regularization_losses
ylayer_metrics
8trainable_variables
9	variables
zlayer_regularization_losses
{metrics
 
 

#0
$1

#0
$1
Ž
|non_trainable_variables

}layers
<regularization_losses
~layer_metrics
=trainable_variables
>	variables
layer_regularization_losses
metrics
 

0
1
 
 
 
 
 

%0
&1

%0
&1
˛
non_trainable_variables
layers
Fregularization_losses
layer_metrics
Gtrainable_variables
H	variables
 layer_regularization_losses
metrics
 
 

'0
(1

'0
(1
˛
non_trainable_variables
layers
Kregularization_losses
layer_metrics
Ltrainable_variables
M	variables
 layer_regularization_losses
metrics
 
 

)0
*1

)0
*1
˛
non_trainable_variables
layers
Pregularization_losses
layer_metrics
Qtrainable_variables
R	variables
 layer_regularization_losses
metrics
 

0
1
2
 
 
 
 
 

+0
,1

+0
,1
˛
non_trainable_variables
layers
Zregularization_losses
layer_metrics
[trainable_variables
\	variables
 layer_regularization_losses
metrics
 
 
 
 
˛
non_trainable_variables
layers
_regularization_losses
layer_metrics
`trainable_variables
a	variables
 layer_regularization_losses
metrics
 
 

-0
.1

-0
.1
˛
non_trainable_variables
layers
dregularization_losses
layer_metrics
etrainable_variables
f	variables
 layer_regularization_losses
metrics
 
 
 
 
˛
non_trainable_variables
 layers
iregularization_losses
Ąlayer_metrics
jtrainable_variables
k	variables
 ˘layer_regularization_losses
Łmetrics
 
 

/0
01

/0
01
˛
¤non_trainable_variables
Ľlayers
nregularization_losses
Ślayer_metrics
otrainable_variables
p	variables
 §layer_regularization_losses
¨metrics
 
#
0
1
2
3
4
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
r
serving_default_input_1Placeholder*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0*
shape:˙˙˙˙˙˙˙˙˙
Ŕ
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1dense_8/kerneldense_8/biasdense_9/kerneldense_9/biasdense_10/kerneldense_10/biasdense_11/kerneldense_11/biasdense_12/kerneldense_12/biasdense_13/kerneldense_13/biasdense_14/kerneldense_14/biasdense_15/kerneldense_15/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *,
f'R%
#__inference_signature_wrapper_54411
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
ć
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename"dense_8/kernel/Read/ReadVariableOp dense_8/bias/Read/ReadVariableOp"dense_9/kernel/Read/ReadVariableOp dense_9/bias/Read/ReadVariableOp#dense_10/kernel/Read/ReadVariableOp!dense_10/bias/Read/ReadVariableOp#dense_11/kernel/Read/ReadVariableOp!dense_11/bias/Read/ReadVariableOp#dense_12/kernel/Read/ReadVariableOp!dense_12/bias/Read/ReadVariableOp#dense_13/kernel/Read/ReadVariableOp!dense_13/bias/Read/ReadVariableOp#dense_14/kernel/Read/ReadVariableOp!dense_14/bias/Read/ReadVariableOp#dense_15/kernel/Read/ReadVariableOp!dense_15/bias/Read/ReadVariableOpConst*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *'
f"R 
__inference__traced_save_55417
Ą
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_8/kerneldense_8/biasdense_9/kerneldense_9/biasdense_10/kerneldense_10/biasdense_11/kerneldense_11/biasdense_12/kerneldense_12/biasdense_13/kerneldense_13/biasdense_14/kerneldense_14/biasdense_15/kerneldense_15/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 **
f%R#
!__inference__traced_restore_55475Â
Ë
b
D__inference_dropout_2_layer_call_and_return_conditional_losses_53878

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity_1"!

identity_1Identity_1:output:0*'
_input_shapes
:˙˙˙˙˙˙˙˙˙:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
ů
Ä
+__inference_link_update_layer_call_fn_53794
dense_10_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity˘StatefulPartitionedCall˛
StatefulPartitionedCallStatefulPartitionedCalldense_10_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_link_update_layer_call_and_return_conditional_losses_537792
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:˙˙˙˙˙˙˙˙˙0::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
'
_output_shapes
:˙˙˙˙˙˙˙˙˙0
(
_user_specified_namedense_10_input
¤
Ť
C__inference_dense_13_layer_call_and_return_conditional_losses_53845

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	@*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2	
BiasAddY
TanhTanhBiasAdd:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
Tanh]
IdentityIdentityTanh:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*.
_input_shapes
:˙˙˙˙˙˙˙˙˙@:::O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙@
 
_user_specified_nameinputs
Ą
Ť
C__inference_dense_11_layer_call_and_return_conditional_losses_53694

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@2	
BiasAddX
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@2
Tanh\
IdentityIdentityTanh:y:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@2

Identity"
identityIdentity:output:0*/
_input_shapes
:˙˙˙˙˙˙˙˙˙:::P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
ˇ
Ć
F__inference_link_update_layer_call_and_return_conditional_losses_53815

inputs
dense_10_53799
dense_10_53801
dense_11_53804
dense_11_53806
dense_12_53809
dense_12_53811
identity˘ dense_10/StatefulPartitionedCall˘ dense_11/StatefulPartitionedCall˘ dense_12/StatefulPartitionedCall
 dense_10/StatefulPartitionedCallStatefulPartitionedCallinputsdense_10_53799dense_10_53801*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_10_layer_call_and_return_conditional_losses_536672"
 dense_10/StatefulPartitionedCall´
 dense_11/StatefulPartitionedCallStatefulPartitionedCall)dense_10/StatefulPartitionedCall:output:0dense_11_53804dense_11_53806*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_11_layer_call_and_return_conditional_losses_536942"
 dense_11/StatefulPartitionedCall´
 dense_12/StatefulPartitionedCallStatefulPartitionedCall)dense_11/StatefulPartitionedCall:output:0dense_12_53809dense_12_53811*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_12_layer_call_and_return_conditional_losses_537212"
 dense_12/StatefulPartitionedCallć
IdentityIdentity)dense_12/StatefulPartitionedCall:output:0!^dense_10/StatefulPartitionedCall!^dense_11/StatefulPartitionedCall!^dense_12/StatefulPartitionedCall*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:˙˙˙˙˙˙˙˙˙0::::::2D
 dense_10/StatefulPartitionedCall dense_10/StatefulPartitionedCall2D
 dense_11/StatefulPartitionedCall dense_11/StatefulPartitionedCall2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall:O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙0
 
_user_specified_nameinputs
ů
Ä
+__inference_link_update_layer_call_fn_53830
dense_10_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity˘StatefulPartitionedCall˛
StatefulPartitionedCallStatefulPartitionedCalldense_10_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_link_update_layer_call_and_return_conditional_losses_538152
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:˙˙˙˙˙˙˙˙˙0::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
'
_output_shapes
:˙˙˙˙˙˙˙˙˙0
(
_user_specified_namedense_10_input
¤
Ť
C__inference_dense_13_layer_call_and_return_conditional_losses_55244

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	@*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2	
BiasAddY
TanhTanhBiasAdd:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
Tanh]
IdentityIdentityTanh:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*.
_input_shapes
:˙˙˙˙˙˙˙˙˙@:::O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙@
 
_user_specified_nameinputs
ľ
¸
'__inference_readout_layer_call_fn_55033

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity˘StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_readout_layer_call_and_return_conditional_losses_541642
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes

:2

Identity"
identityIdentity:output:0*5
_input_shapes$
":@::::::22
StatefulPartitionedCallStatefulPartitionedCall:F B

_output_shapes

:@
 
_user_specified_nameinputs
Ź°

!__inference_message_passing_44574	
input9
5create_message_dense_8_matmul_readvariableop_resource:
6create_message_dense_8_biasadd_readvariableop_resource9
5create_message_dense_9_matmul_readvariableop_resource:
6create_message_dense_9_biasadd_readvariableop_resource7
3link_update_dense_10_matmul_readvariableop_resource8
4link_update_dense_10_biasadd_readvariableop_resource7
3link_update_dense_11_matmul_readvariableop_resource8
4link_update_dense_11_biasadd_readvariableop_resource7
3link_update_dense_12_matmul_readvariableop_resource8
4link_update_dense_12_biasadd_readvariableop_resource
identityo
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"   J   2
Reshape/shapee
ReshapeReshapeinputReshape/shape:output:0*
T0*
_output_shapes

:J2	
Reshapeq
transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose/permw
	transpose	TransposeReshape:output:0transpose/perm:output:0*
T0*
_output_shapes

:J2
	transpose}
Pad/paddingsConst*
_output_shapes

:*
dtype0*)
value B"               2
Pad/paddings`
PadPadtranspose:y:0Pad/paddings:output:0*
T0*
_output_shapes

:J2
Pad	
GatherV2/indicesConst*
_output_shapes	
:*
dtype0*ą
value§B¤"                                                      
   
   
   
   
                                                                                                                                            !   !   !   !   !   !   "   "   "   "   "   "   "   "   #   #   #   %   %   %   %   %   %   &   &   &   '   '   '   '   '   '   '   '   (   (   (   (   )   )   )   )   )   )   )   )   *   *   +   +   +   +   +   +   ,   ,   ,   ,   -   -   -   -   -   -   -   -   .   .   .   .   /   /   /   0   0   0   0   0   1   1   1   1   1   1   1   1   2   2   2   2   3   3   3   3   4   4   4   4   4   5   5   5   6   6   6   6   7   7   7   8   8   8   8   8   8   9   9   9   9   9   9   9   9   :   :   :   :   ;   ;   ;   ;   <   <   =   =   =   =   >   >   >   >   ?   ?   ?   ?   ?   @   @   A   A   A   A   A   A   B   B   B   B   C   C   C   C   C   C   C   C   D   D   D   D   E   E   E   E   F   F   F   F   G   G   G   G   H   H   H   H   H   I   I   I   I   I   I   I   I   2
GatherV2/indices`
GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2/axisŽ
GatherV2GatherV2Pad:output:0GatherV2/indices:output:0GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:	2

GatherV2	
GatherV2_1/indicesConst*
_output_shapes	
:*
dtype0*ą
value§B¤"      
                                  #   %   &   '   5   6   7   8   9          D   E   F   G   H   I                                                                         !   "                                 B   C   D   E   F   G   H   I   :   ;   <   =   >   ?   @   A         
   D   E   F   G   H   I   2   3   4   :   ;   <   =   >   ?   @   A   *   +   ,   -   :   ;   <   =   >   ?   @   A   (   )   D   E   F   G   H   I   .   /   0   1   :   ;   <   =   >   ?   @   A   *   +   ,   -   2   3   4   5   6   7   8   9   :   ;   <   =   >   ?   @   A   #   %   &   '   .   /   0   1   5   6   7   8   9         
   .   /   0   1   2   3   4   D   E   F   G   H   I   :   ;   <   =   >   ?   @   A          !   "   #   %   &   '   (   )   *   +   ,   -   .   /   0   1   5   6   7   8   9   B   C   D   E   F   G   H   I          !   "   :   ;   <   =   >   ?   @   A                      !   "   #   %   &   '   *   +   ,   -   5   6   7   8   9   :   ;   <   =   >   ?   @   A   2
GatherV2_1/indicesd
GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2_1/axisś

GatherV2_1GatherV2Pad:output:0GatherV2_1/indices:output:0GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:	2

GatherV2_1\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis
concatConcatV2GatherV2:output:0GatherV2_1:output:0concat/axis:output:0*
N*
T0*
_output_shapes
:	 2
concatŇ
,create_message/dense_8/MatMul/ReadVariableOpReadVariableOp5create_message_dense_8_matmul_readvariableop_resource*
_output_shapes

: @*
dtype02.
,create_message/dense_8/MatMul/ReadVariableOpš
create_message/dense_8/MatMulMatMulconcat:output:04create_message/dense_8/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	@2
create_message/dense_8/MatMulŃ
-create_message/dense_8/BiasAdd/ReadVariableOpReadVariableOp6create_message_dense_8_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-create_message/dense_8/BiasAdd/ReadVariableOpŐ
create_message/dense_8/BiasAddBiasAdd'create_message/dense_8/MatMul:product:05create_message/dense_8/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	@2 
create_message/dense_8/BiasAdd
create_message/dense_8/TanhTanh'create_message/dense_8/BiasAdd:output:0*
T0*
_output_shapes
:	@2
create_message/dense_8/TanhŇ
,create_message/dense_9/MatMul/ReadVariableOpReadVariableOp5create_message_dense_9_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02.
,create_message/dense_9/MatMul/ReadVariableOpÉ
create_message/dense_9/MatMulMatMulcreate_message/dense_8/Tanh:y:04create_message/dense_9/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
create_message/dense_9/MatMulŃ
-create_message/dense_9/BiasAdd/ReadVariableOpReadVariableOp6create_message_dense_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-create_message/dense_9/BiasAdd/ReadVariableOpŐ
create_message/dense_9/BiasAddBiasAdd'create_message/dense_9/MatMul:product:05create_message/dense_9/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	2 
create_message/dense_9/BiasAdd
create_message/dense_9/TanhTanh'create_message/dense_9/BiasAdd:output:0*
T0*
_output_shapes
:	2
create_message/dense_9/Tanh˛
PartitionedCallPartitionedCallcreate_message/dense_9/Tanh:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:J * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *-
f(R&
$__inference_message_aggregation_12562
PartitionedCall`
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat_1/axis
concat_1ConcatV2Pad:output:0PartitionedCall:output:0concat_1/axis:output:0*
N*
T0*
_output_shapes

:J02

concat_1Í
*link_update/dense_10/MatMul/ReadVariableOpReadVariableOp3link_update_dense_10_matmul_readvariableop_resource*
_output_shapes
:	0*
dtype02,
*link_update/dense_10/MatMul/ReadVariableOpľ
link_update/dense_10/MatMulMatMulconcat_1:output:02link_update/dense_10/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	J2
link_update/dense_10/MatMulĚ
+link_update/dense_10/BiasAdd/ReadVariableOpReadVariableOp4link_update_dense_10_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02-
+link_update/dense_10/BiasAdd/ReadVariableOpÍ
link_update/dense_10/BiasAddBiasAdd%link_update/dense_10/MatMul:product:03link_update/dense_10/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	J2
link_update/dense_10/BiasAdd
link_update/dense_10/TanhTanh%link_update/dense_10/BiasAdd:output:0*
T0*
_output_shapes
:	J2
link_update/dense_10/TanhÍ
*link_update/dense_11/MatMul/ReadVariableOpReadVariableOp3link_update_dense_11_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype02,
*link_update/dense_11/MatMul/ReadVariableOpŔ
link_update/dense_11/MatMulMatMullink_update/dense_10/Tanh:y:02link_update/dense_11/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:J@2
link_update/dense_11/MatMulË
+link_update/dense_11/BiasAdd/ReadVariableOpReadVariableOp4link_update_dense_11_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02-
+link_update/dense_11/BiasAdd/ReadVariableOpĚ
link_update/dense_11/BiasAddBiasAdd%link_update/dense_11/MatMul:product:03link_update/dense_11/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:J@2
link_update/dense_11/BiasAdd
link_update/dense_11/TanhTanh%link_update/dense_11/BiasAdd:output:0*
T0*
_output_shapes

:J@2
link_update/dense_11/TanhĚ
*link_update/dense_12/MatMul/ReadVariableOpReadVariableOp3link_update_dense_12_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02,
*link_update/dense_12/MatMul/ReadVariableOpŔ
link_update/dense_12/MatMulMatMullink_update/dense_11/Tanh:y:02link_update/dense_12/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:J2
link_update/dense_12/MatMulË
+link_update/dense_12/BiasAdd/ReadVariableOpReadVariableOp4link_update_dense_12_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02-
+link_update/dense_12/BiasAdd/ReadVariableOpĚ
link_update/dense_12/BiasAddBiasAdd%link_update/dense_12/MatMul:product:03link_update/dense_12/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:J2
link_update/dense_12/BiasAdd
link_update/dense_12/TanhTanh%link_update/dense_12/BiasAdd:output:0*
T0*
_output_shapes

:J2
link_update/dense_12/Tanh	
GatherV2_2/indicesConst*
_output_shapes	
:*
dtype0*ą
value§B¤"                                                      
   
   
   
   
                                                                                                                                            !   !   !   !   !   !   "   "   "   "   "   "   "   "   #   #   #   %   %   %   %   %   %   &   &   &   '   '   '   '   '   '   '   '   (   (   (   (   )   )   )   )   )   )   )   )   *   *   +   +   +   +   +   +   ,   ,   ,   ,   -   -   -   -   -   -   -   -   .   .   .   .   /   /   /   0   0   0   0   0   1   1   1   1   1   1   1   1   2   2   2   2   3   3   3   3   4   4   4   4   4   5   5   5   6   6   6   6   7   7   7   8   8   8   8   8   8   9   9   9   9   9   9   9   9   :   :   :   :   ;   ;   ;   ;   <   <   =   =   =   =   >   >   >   >   ?   ?   ?   ?   ?   @   @   A   A   A   A   A   A   B   B   B   B   C   C   C   C   C   C   C   C   D   D   D   D   E   E   E   E   F   F   F   F   G   G   G   G   H   H   H   H   H   I   I   I   I   I   I   I   I   2
GatherV2_2/indicesd
GatherV2_2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2_2/axisÇ

GatherV2_2GatherV2link_update/dense_12/Tanh:y:0GatherV2_2/indices:output:0GatherV2_2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:	2

GatherV2_2	
GatherV2_3/indicesConst*
_output_shapes	
:*
dtype0*ą
value§B¤"      
                                  #   %   &   '   5   6   7   8   9          D   E   F   G   H   I                                                                         !   "                                 B   C   D   E   F   G   H   I   :   ;   <   =   >   ?   @   A         
   D   E   F   G   H   I   2   3   4   :   ;   <   =   >   ?   @   A   *   +   ,   -   :   ;   <   =   >   ?   @   A   (   )   D   E   F   G   H   I   .   /   0   1   :   ;   <   =   >   ?   @   A   *   +   ,   -   2   3   4   5   6   7   8   9   :   ;   <   =   >   ?   @   A   #   %   &   '   .   /   0   1   5   6   7   8   9         
   .   /   0   1   2   3   4   D   E   F   G   H   I   :   ;   <   =   >   ?   @   A          !   "   #   %   &   '   (   )   *   +   ,   -   .   /   0   1   5   6   7   8   9   B   C   D   E   F   G   H   I          !   "   :   ;   <   =   >   ?   @   A                      !   "   #   %   &   '   *   +   ,   -   5   6   7   8   9   :   ;   <   =   >   ?   @   A   2
GatherV2_3/indicesd
GatherV2_3/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2_3/axisÇ

GatherV2_3GatherV2link_update/dense_12/Tanh:y:0GatherV2_3/indices:output:0GatherV2_3/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:	2

GatherV2_3`
concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat_2/axis
concat_2ConcatV2GatherV2_2:output:0GatherV2_3:output:0concat_2/axis:output:0*
N*
T0*
_output_shapes
:	 2

concat_2Ö
.create_message/dense_8/MatMul_1/ReadVariableOpReadVariableOp5create_message_dense_8_matmul_readvariableop_resource*
_output_shapes

: @*
dtype020
.create_message/dense_8/MatMul_1/ReadVariableOpÁ
create_message/dense_8/MatMul_1MatMulconcat_2:output:06create_message/dense_8/MatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	@2!
create_message/dense_8/MatMul_1Ő
/create_message/dense_8/BiasAdd_1/ReadVariableOpReadVariableOp6create_message_dense_8_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype021
/create_message/dense_8/BiasAdd_1/ReadVariableOpÝ
 create_message/dense_8/BiasAdd_1BiasAdd)create_message/dense_8/MatMul_1:product:07create_message/dense_8/BiasAdd_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	@2"
 create_message/dense_8/BiasAdd_1
create_message/dense_8/Tanh_1Tanh)create_message/dense_8/BiasAdd_1:output:0*
T0*
_output_shapes
:	@2
create_message/dense_8/Tanh_1Ö
.create_message/dense_9/MatMul_1/ReadVariableOpReadVariableOp5create_message_dense_9_matmul_readvariableop_resource*
_output_shapes

:@*
dtype020
.create_message/dense_9/MatMul_1/ReadVariableOpŃ
create_message/dense_9/MatMul_1MatMul!create_message/dense_8/Tanh_1:y:06create_message/dense_9/MatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	2!
create_message/dense_9/MatMul_1Ő
/create_message/dense_9/BiasAdd_1/ReadVariableOpReadVariableOp6create_message_dense_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype021
/create_message/dense_9/BiasAdd_1/ReadVariableOpÝ
 create_message/dense_9/BiasAdd_1BiasAdd)create_message/dense_9/MatMul_1:product:07create_message/dense_9/BiasAdd_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	2"
 create_message/dense_9/BiasAdd_1
create_message/dense_9/Tanh_1Tanh)create_message/dense_9/BiasAdd_1:output:0*
T0*
_output_shapes
:	2
create_message/dense_9/Tanh_1¸
PartitionedCall_1PartitionedCall!create_message/dense_9/Tanh_1:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:J * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *-
f(R&
$__inference_message_aggregation_12562
PartitionedCall_1`
concat_3/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat_3/axisĽ
concat_3ConcatV2link_update/dense_12/Tanh:y:0PartitionedCall_1:output:0concat_3/axis:output:0*
N*
T0*
_output_shapes

:J02

concat_3Ń
,link_update/dense_10/MatMul_1/ReadVariableOpReadVariableOp3link_update_dense_10_matmul_readvariableop_resource*
_output_shapes
:	0*
dtype02.
,link_update/dense_10/MatMul_1/ReadVariableOpť
link_update/dense_10/MatMul_1MatMulconcat_3:output:04link_update/dense_10/MatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	J2
link_update/dense_10/MatMul_1Đ
-link_update/dense_10/BiasAdd_1/ReadVariableOpReadVariableOp4link_update_dense_10_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02/
-link_update/dense_10/BiasAdd_1/ReadVariableOpŐ
link_update/dense_10/BiasAdd_1BiasAdd'link_update/dense_10/MatMul_1:product:05link_update/dense_10/BiasAdd_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	J2 
link_update/dense_10/BiasAdd_1
link_update/dense_10/Tanh_1Tanh'link_update/dense_10/BiasAdd_1:output:0*
T0*
_output_shapes
:	J2
link_update/dense_10/Tanh_1Ń
,link_update/dense_11/MatMul_1/ReadVariableOpReadVariableOp3link_update_dense_11_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype02.
,link_update/dense_11/MatMul_1/ReadVariableOpČ
link_update/dense_11/MatMul_1MatMullink_update/dense_10/Tanh_1:y:04link_update/dense_11/MatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes

:J@2
link_update/dense_11/MatMul_1Ď
-link_update/dense_11/BiasAdd_1/ReadVariableOpReadVariableOp4link_update_dense_11_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-link_update/dense_11/BiasAdd_1/ReadVariableOpÔ
link_update/dense_11/BiasAdd_1BiasAdd'link_update/dense_11/MatMul_1:product:05link_update/dense_11/BiasAdd_1/ReadVariableOp:value:0*
T0*
_output_shapes

:J@2 
link_update/dense_11/BiasAdd_1
link_update/dense_11/Tanh_1Tanh'link_update/dense_11/BiasAdd_1:output:0*
T0*
_output_shapes

:J@2
link_update/dense_11/Tanh_1Đ
,link_update/dense_12/MatMul_1/ReadVariableOpReadVariableOp3link_update_dense_12_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02.
,link_update/dense_12/MatMul_1/ReadVariableOpČ
link_update/dense_12/MatMul_1MatMullink_update/dense_11/Tanh_1:y:04link_update/dense_12/MatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes

:J2
link_update/dense_12/MatMul_1Ď
-link_update/dense_12/BiasAdd_1/ReadVariableOpReadVariableOp4link_update_dense_12_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-link_update/dense_12/BiasAdd_1/ReadVariableOpÔ
link_update/dense_12/BiasAdd_1BiasAdd'link_update/dense_12/MatMul_1:product:05link_update/dense_12/BiasAdd_1/ReadVariableOp:value:0*
T0*
_output_shapes

:J2 
link_update/dense_12/BiasAdd_1
link_update/dense_12/Tanh_1Tanh'link_update/dense_12/BiasAdd_1:output:0*
T0*
_output_shapes

:J2
link_update/dense_12/Tanh_1	
GatherV2_4/indicesConst*
_output_shapes	
:*
dtype0*ą
value§B¤"                                                      
   
   
   
   
                                                                                                                                            !   !   !   !   !   !   "   "   "   "   "   "   "   "   #   #   #   %   %   %   %   %   %   &   &   &   '   '   '   '   '   '   '   '   (   (   (   (   )   )   )   )   )   )   )   )   *   *   +   +   +   +   +   +   ,   ,   ,   ,   -   -   -   -   -   -   -   -   .   .   .   .   /   /   /   0   0   0   0   0   1   1   1   1   1   1   1   1   2   2   2   2   3   3   3   3   4   4   4   4   4   5   5   5   6   6   6   6   7   7   7   8   8   8   8   8   8   9   9   9   9   9   9   9   9   :   :   :   :   ;   ;   ;   ;   <   <   =   =   =   =   >   >   >   >   ?   ?   ?   ?   ?   @   @   A   A   A   A   A   A   B   B   B   B   C   C   C   C   C   C   C   C   D   D   D   D   E   E   E   E   F   F   F   F   G   G   G   G   H   H   H   H   H   I   I   I   I   I   I   I   I   2
GatherV2_4/indicesd
GatherV2_4/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2_4/axisÉ

GatherV2_4GatherV2link_update/dense_12/Tanh_1:y:0GatherV2_4/indices:output:0GatherV2_4/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:	2

GatherV2_4	
GatherV2_5/indicesConst*
_output_shapes	
:*
dtype0*ą
value§B¤"      
                                  #   %   &   '   5   6   7   8   9          D   E   F   G   H   I                                                                         !   "                                 B   C   D   E   F   G   H   I   :   ;   <   =   >   ?   @   A         
   D   E   F   G   H   I   2   3   4   :   ;   <   =   >   ?   @   A   *   +   ,   -   :   ;   <   =   >   ?   @   A   (   )   D   E   F   G   H   I   .   /   0   1   :   ;   <   =   >   ?   @   A   *   +   ,   -   2   3   4   5   6   7   8   9   :   ;   <   =   >   ?   @   A   #   %   &   '   .   /   0   1   5   6   7   8   9         
   .   /   0   1   2   3   4   D   E   F   G   H   I   :   ;   <   =   >   ?   @   A          !   "   #   %   &   '   (   )   *   +   ,   -   .   /   0   1   5   6   7   8   9   B   C   D   E   F   G   H   I          !   "   :   ;   <   =   >   ?   @   A                      !   "   #   %   &   '   *   +   ,   -   5   6   7   8   9   :   ;   <   =   >   ?   @   A   2
GatherV2_5/indicesd
GatherV2_5/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2_5/axisÉ

GatherV2_5GatherV2link_update/dense_12/Tanh_1:y:0GatherV2_5/indices:output:0GatherV2_5/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:	2

GatherV2_5`
concat_4/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat_4/axis
concat_4ConcatV2GatherV2_4:output:0GatherV2_5:output:0concat_4/axis:output:0*
N*
T0*
_output_shapes
:	 2

concat_4Ö
.create_message/dense_8/MatMul_2/ReadVariableOpReadVariableOp5create_message_dense_8_matmul_readvariableop_resource*
_output_shapes

: @*
dtype020
.create_message/dense_8/MatMul_2/ReadVariableOpÁ
create_message/dense_8/MatMul_2MatMulconcat_4:output:06create_message/dense_8/MatMul_2/ReadVariableOp:value:0*
T0*
_output_shapes
:	@2!
create_message/dense_8/MatMul_2Ő
/create_message/dense_8/BiasAdd_2/ReadVariableOpReadVariableOp6create_message_dense_8_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype021
/create_message/dense_8/BiasAdd_2/ReadVariableOpÝ
 create_message/dense_8/BiasAdd_2BiasAdd)create_message/dense_8/MatMul_2:product:07create_message/dense_8/BiasAdd_2/ReadVariableOp:value:0*
T0*
_output_shapes
:	@2"
 create_message/dense_8/BiasAdd_2
create_message/dense_8/Tanh_2Tanh)create_message/dense_8/BiasAdd_2:output:0*
T0*
_output_shapes
:	@2
create_message/dense_8/Tanh_2Ö
.create_message/dense_9/MatMul_2/ReadVariableOpReadVariableOp5create_message_dense_9_matmul_readvariableop_resource*
_output_shapes

:@*
dtype020
.create_message/dense_9/MatMul_2/ReadVariableOpŃ
create_message/dense_9/MatMul_2MatMul!create_message/dense_8/Tanh_2:y:06create_message/dense_9/MatMul_2/ReadVariableOp:value:0*
T0*
_output_shapes
:	2!
create_message/dense_9/MatMul_2Ő
/create_message/dense_9/BiasAdd_2/ReadVariableOpReadVariableOp6create_message_dense_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype021
/create_message/dense_9/BiasAdd_2/ReadVariableOpÝ
 create_message/dense_9/BiasAdd_2BiasAdd)create_message/dense_9/MatMul_2:product:07create_message/dense_9/BiasAdd_2/ReadVariableOp:value:0*
T0*
_output_shapes
:	2"
 create_message/dense_9/BiasAdd_2
create_message/dense_9/Tanh_2Tanh)create_message/dense_9/BiasAdd_2:output:0*
T0*
_output_shapes
:	2
create_message/dense_9/Tanh_2¸
PartitionedCall_2PartitionedCall!create_message/dense_9/Tanh_2:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:J * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *-
f(R&
$__inference_message_aggregation_12562
PartitionedCall_2`
concat_5/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat_5/axis§
concat_5ConcatV2link_update/dense_12/Tanh_1:y:0PartitionedCall_2:output:0concat_5/axis:output:0*
N*
T0*
_output_shapes

:J02

concat_5Ń
,link_update/dense_10/MatMul_2/ReadVariableOpReadVariableOp3link_update_dense_10_matmul_readvariableop_resource*
_output_shapes
:	0*
dtype02.
,link_update/dense_10/MatMul_2/ReadVariableOpť
link_update/dense_10/MatMul_2MatMulconcat_5:output:04link_update/dense_10/MatMul_2/ReadVariableOp:value:0*
T0*
_output_shapes
:	J2
link_update/dense_10/MatMul_2Đ
-link_update/dense_10/BiasAdd_2/ReadVariableOpReadVariableOp4link_update_dense_10_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02/
-link_update/dense_10/BiasAdd_2/ReadVariableOpŐ
link_update/dense_10/BiasAdd_2BiasAdd'link_update/dense_10/MatMul_2:product:05link_update/dense_10/BiasAdd_2/ReadVariableOp:value:0*
T0*
_output_shapes
:	J2 
link_update/dense_10/BiasAdd_2
link_update/dense_10/Tanh_2Tanh'link_update/dense_10/BiasAdd_2:output:0*
T0*
_output_shapes
:	J2
link_update/dense_10/Tanh_2Ń
,link_update/dense_11/MatMul_2/ReadVariableOpReadVariableOp3link_update_dense_11_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype02.
,link_update/dense_11/MatMul_2/ReadVariableOpČ
link_update/dense_11/MatMul_2MatMullink_update/dense_10/Tanh_2:y:04link_update/dense_11/MatMul_2/ReadVariableOp:value:0*
T0*
_output_shapes

:J@2
link_update/dense_11/MatMul_2Ď
-link_update/dense_11/BiasAdd_2/ReadVariableOpReadVariableOp4link_update_dense_11_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-link_update/dense_11/BiasAdd_2/ReadVariableOpÔ
link_update/dense_11/BiasAdd_2BiasAdd'link_update/dense_11/MatMul_2:product:05link_update/dense_11/BiasAdd_2/ReadVariableOp:value:0*
T0*
_output_shapes

:J@2 
link_update/dense_11/BiasAdd_2
link_update/dense_11/Tanh_2Tanh'link_update/dense_11/BiasAdd_2:output:0*
T0*
_output_shapes

:J@2
link_update/dense_11/Tanh_2Đ
,link_update/dense_12/MatMul_2/ReadVariableOpReadVariableOp3link_update_dense_12_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02.
,link_update/dense_12/MatMul_2/ReadVariableOpČ
link_update/dense_12/MatMul_2MatMullink_update/dense_11/Tanh_2:y:04link_update/dense_12/MatMul_2/ReadVariableOp:value:0*
T0*
_output_shapes

:J2
link_update/dense_12/MatMul_2Ď
-link_update/dense_12/BiasAdd_2/ReadVariableOpReadVariableOp4link_update_dense_12_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-link_update/dense_12/BiasAdd_2/ReadVariableOpÔ
link_update/dense_12/BiasAdd_2BiasAdd'link_update/dense_12/MatMul_2:product:05link_update/dense_12/BiasAdd_2/ReadVariableOp:value:0*
T0*
_output_shapes

:J2 
link_update/dense_12/BiasAdd_2
link_update/dense_12/Tanh_2Tanh'link_update/dense_12/BiasAdd_2:output:0*
T0*
_output_shapes

:J2
link_update/dense_12/Tanh_2	
GatherV2_6/indicesConst*
_output_shapes	
:*
dtype0*ą
value§B¤"                                                      
   
   
   
   
                                                                                                                                            !   !   !   !   !   !   "   "   "   "   "   "   "   "   #   #   #   %   %   %   %   %   %   &   &   &   '   '   '   '   '   '   '   '   (   (   (   (   )   )   )   )   )   )   )   )   *   *   +   +   +   +   +   +   ,   ,   ,   ,   -   -   -   -   -   -   -   -   .   .   .   .   /   /   /   0   0   0   0   0   1   1   1   1   1   1   1   1   2   2   2   2   3   3   3   3   4   4   4   4   4   5   5   5   6   6   6   6   7   7   7   8   8   8   8   8   8   9   9   9   9   9   9   9   9   :   :   :   :   ;   ;   ;   ;   <   <   =   =   =   =   >   >   >   >   ?   ?   ?   ?   ?   @   @   A   A   A   A   A   A   B   B   B   B   C   C   C   C   C   C   C   C   D   D   D   D   E   E   E   E   F   F   F   F   G   G   G   G   H   H   H   H   H   I   I   I   I   I   I   I   I   2
GatherV2_6/indicesd
GatherV2_6/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2_6/axisÉ

GatherV2_6GatherV2link_update/dense_12/Tanh_2:y:0GatherV2_6/indices:output:0GatherV2_6/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:	2

GatherV2_6	
GatherV2_7/indicesConst*
_output_shapes	
:*
dtype0*ą
value§B¤"      
                                  #   %   &   '   5   6   7   8   9          D   E   F   G   H   I                                                                         !   "                                 B   C   D   E   F   G   H   I   :   ;   <   =   >   ?   @   A         
   D   E   F   G   H   I   2   3   4   :   ;   <   =   >   ?   @   A   *   +   ,   -   :   ;   <   =   >   ?   @   A   (   )   D   E   F   G   H   I   .   /   0   1   :   ;   <   =   >   ?   @   A   *   +   ,   -   2   3   4   5   6   7   8   9   :   ;   <   =   >   ?   @   A   #   %   &   '   .   /   0   1   5   6   7   8   9         
   .   /   0   1   2   3   4   D   E   F   G   H   I   :   ;   <   =   >   ?   @   A          !   "   #   %   &   '   (   )   *   +   ,   -   .   /   0   1   5   6   7   8   9   B   C   D   E   F   G   H   I          !   "   :   ;   <   =   >   ?   @   A                      !   "   #   %   &   '   *   +   ,   -   5   6   7   8   9   :   ;   <   =   >   ?   @   A   2
GatherV2_7/indicesd
GatherV2_7/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2_7/axisÉ

GatherV2_7GatherV2link_update/dense_12/Tanh_2:y:0GatherV2_7/indices:output:0GatherV2_7/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:	2

GatherV2_7`
concat_6/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat_6/axis
concat_6ConcatV2GatherV2_6:output:0GatherV2_7:output:0concat_6/axis:output:0*
N*
T0*
_output_shapes
:	 2

concat_6Ö
.create_message/dense_8/MatMul_3/ReadVariableOpReadVariableOp5create_message_dense_8_matmul_readvariableop_resource*
_output_shapes

: @*
dtype020
.create_message/dense_8/MatMul_3/ReadVariableOpÁ
create_message/dense_8/MatMul_3MatMulconcat_6:output:06create_message/dense_8/MatMul_3/ReadVariableOp:value:0*
T0*
_output_shapes
:	@2!
create_message/dense_8/MatMul_3Ő
/create_message/dense_8/BiasAdd_3/ReadVariableOpReadVariableOp6create_message_dense_8_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype021
/create_message/dense_8/BiasAdd_3/ReadVariableOpÝ
 create_message/dense_8/BiasAdd_3BiasAdd)create_message/dense_8/MatMul_3:product:07create_message/dense_8/BiasAdd_3/ReadVariableOp:value:0*
T0*
_output_shapes
:	@2"
 create_message/dense_8/BiasAdd_3
create_message/dense_8/Tanh_3Tanh)create_message/dense_8/BiasAdd_3:output:0*
T0*
_output_shapes
:	@2
create_message/dense_8/Tanh_3Ö
.create_message/dense_9/MatMul_3/ReadVariableOpReadVariableOp5create_message_dense_9_matmul_readvariableop_resource*
_output_shapes

:@*
dtype020
.create_message/dense_9/MatMul_3/ReadVariableOpŃ
create_message/dense_9/MatMul_3MatMul!create_message/dense_8/Tanh_3:y:06create_message/dense_9/MatMul_3/ReadVariableOp:value:0*
T0*
_output_shapes
:	2!
create_message/dense_9/MatMul_3Ő
/create_message/dense_9/BiasAdd_3/ReadVariableOpReadVariableOp6create_message_dense_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype021
/create_message/dense_9/BiasAdd_3/ReadVariableOpÝ
 create_message/dense_9/BiasAdd_3BiasAdd)create_message/dense_9/MatMul_3:product:07create_message/dense_9/BiasAdd_3/ReadVariableOp:value:0*
T0*
_output_shapes
:	2"
 create_message/dense_9/BiasAdd_3
create_message/dense_9/Tanh_3Tanh)create_message/dense_9/BiasAdd_3:output:0*
T0*
_output_shapes
:	2
create_message/dense_9/Tanh_3¸
PartitionedCall_3PartitionedCall!create_message/dense_9/Tanh_3:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:J * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *-
f(R&
$__inference_message_aggregation_12562
PartitionedCall_3`
concat_7/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat_7/axis§
concat_7ConcatV2link_update/dense_12/Tanh_2:y:0PartitionedCall_3:output:0concat_7/axis:output:0*
N*
T0*
_output_shapes

:J02

concat_7Ń
,link_update/dense_10/MatMul_3/ReadVariableOpReadVariableOp3link_update_dense_10_matmul_readvariableop_resource*
_output_shapes
:	0*
dtype02.
,link_update/dense_10/MatMul_3/ReadVariableOpť
link_update/dense_10/MatMul_3MatMulconcat_7:output:04link_update/dense_10/MatMul_3/ReadVariableOp:value:0*
T0*
_output_shapes
:	J2
link_update/dense_10/MatMul_3Đ
-link_update/dense_10/BiasAdd_3/ReadVariableOpReadVariableOp4link_update_dense_10_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02/
-link_update/dense_10/BiasAdd_3/ReadVariableOpŐ
link_update/dense_10/BiasAdd_3BiasAdd'link_update/dense_10/MatMul_3:product:05link_update/dense_10/BiasAdd_3/ReadVariableOp:value:0*
T0*
_output_shapes
:	J2 
link_update/dense_10/BiasAdd_3
link_update/dense_10/Tanh_3Tanh'link_update/dense_10/BiasAdd_3:output:0*
T0*
_output_shapes
:	J2
link_update/dense_10/Tanh_3Ń
,link_update/dense_11/MatMul_3/ReadVariableOpReadVariableOp3link_update_dense_11_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype02.
,link_update/dense_11/MatMul_3/ReadVariableOpČ
link_update/dense_11/MatMul_3MatMullink_update/dense_10/Tanh_3:y:04link_update/dense_11/MatMul_3/ReadVariableOp:value:0*
T0*
_output_shapes

:J@2
link_update/dense_11/MatMul_3Ď
-link_update/dense_11/BiasAdd_3/ReadVariableOpReadVariableOp4link_update_dense_11_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-link_update/dense_11/BiasAdd_3/ReadVariableOpÔ
link_update/dense_11/BiasAdd_3BiasAdd'link_update/dense_11/MatMul_3:product:05link_update/dense_11/BiasAdd_3/ReadVariableOp:value:0*
T0*
_output_shapes

:J@2 
link_update/dense_11/BiasAdd_3
link_update/dense_11/Tanh_3Tanh'link_update/dense_11/BiasAdd_3:output:0*
T0*
_output_shapes

:J@2
link_update/dense_11/Tanh_3Đ
,link_update/dense_12/MatMul_3/ReadVariableOpReadVariableOp3link_update_dense_12_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02.
,link_update/dense_12/MatMul_3/ReadVariableOpČ
link_update/dense_12/MatMul_3MatMullink_update/dense_11/Tanh_3:y:04link_update/dense_12/MatMul_3/ReadVariableOp:value:0*
T0*
_output_shapes

:J2
link_update/dense_12/MatMul_3Ď
-link_update/dense_12/BiasAdd_3/ReadVariableOpReadVariableOp4link_update_dense_12_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-link_update/dense_12/BiasAdd_3/ReadVariableOpÔ
link_update/dense_12/BiasAdd_3BiasAdd'link_update/dense_12/MatMul_3:product:05link_update/dense_12/BiasAdd_3/ReadVariableOp:value:0*
T0*
_output_shapes

:J2 
link_update/dense_12/BiasAdd_3
link_update/dense_12/Tanh_3Tanh'link_update/dense_12/BiasAdd_3:output:0*
T0*
_output_shapes

:J2
link_update/dense_12/Tanh_3	
GatherV2_8/indicesConst*
_output_shapes	
:*
dtype0*ą
value§B¤"                                                      
   
   
   
   
                                                                                                                                            !   !   !   !   !   !   "   "   "   "   "   "   "   "   #   #   #   %   %   %   %   %   %   &   &   &   '   '   '   '   '   '   '   '   (   (   (   (   )   )   )   )   )   )   )   )   *   *   +   +   +   +   +   +   ,   ,   ,   ,   -   -   -   -   -   -   -   -   .   .   .   .   /   /   /   0   0   0   0   0   1   1   1   1   1   1   1   1   2   2   2   2   3   3   3   3   4   4   4   4   4   5   5   5   6   6   6   6   7   7   7   8   8   8   8   8   8   9   9   9   9   9   9   9   9   :   :   :   :   ;   ;   ;   ;   <   <   =   =   =   =   >   >   >   >   ?   ?   ?   ?   ?   @   @   A   A   A   A   A   A   B   B   B   B   C   C   C   C   C   C   C   C   D   D   D   D   E   E   E   E   F   F   F   F   G   G   G   G   H   H   H   H   H   I   I   I   I   I   I   I   I   2
GatherV2_8/indicesd
GatherV2_8/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2_8/axisÉ

GatherV2_8GatherV2link_update/dense_12/Tanh_3:y:0GatherV2_8/indices:output:0GatherV2_8/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:	2

GatherV2_8	
GatherV2_9/indicesConst*
_output_shapes	
:*
dtype0*ą
value§B¤"      
                                  #   %   &   '   5   6   7   8   9          D   E   F   G   H   I                                                                         !   "                                 B   C   D   E   F   G   H   I   :   ;   <   =   >   ?   @   A         
   D   E   F   G   H   I   2   3   4   :   ;   <   =   >   ?   @   A   *   +   ,   -   :   ;   <   =   >   ?   @   A   (   )   D   E   F   G   H   I   .   /   0   1   :   ;   <   =   >   ?   @   A   *   +   ,   -   2   3   4   5   6   7   8   9   :   ;   <   =   >   ?   @   A   #   %   &   '   .   /   0   1   5   6   7   8   9         
   .   /   0   1   2   3   4   D   E   F   G   H   I   :   ;   <   =   >   ?   @   A          !   "   #   %   &   '   (   )   *   +   ,   -   .   /   0   1   5   6   7   8   9   B   C   D   E   F   G   H   I          !   "   :   ;   <   =   >   ?   @   A                      !   "   #   %   &   '   *   +   ,   -   5   6   7   8   9   :   ;   <   =   >   ?   @   A   2
GatherV2_9/indicesd
GatherV2_9/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2_9/axisÉ

GatherV2_9GatherV2link_update/dense_12/Tanh_3:y:0GatherV2_9/indices:output:0GatherV2_9/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:	2

GatherV2_9`
concat_8/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat_8/axis
concat_8ConcatV2GatherV2_8:output:0GatherV2_9:output:0concat_8/axis:output:0*
N*
T0*
_output_shapes
:	 2

concat_8Ö
.create_message/dense_8/MatMul_4/ReadVariableOpReadVariableOp5create_message_dense_8_matmul_readvariableop_resource*
_output_shapes

: @*
dtype020
.create_message/dense_8/MatMul_4/ReadVariableOpÁ
create_message/dense_8/MatMul_4MatMulconcat_8:output:06create_message/dense_8/MatMul_4/ReadVariableOp:value:0*
T0*
_output_shapes
:	@2!
create_message/dense_8/MatMul_4Ő
/create_message/dense_8/BiasAdd_4/ReadVariableOpReadVariableOp6create_message_dense_8_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype021
/create_message/dense_8/BiasAdd_4/ReadVariableOpÝ
 create_message/dense_8/BiasAdd_4BiasAdd)create_message/dense_8/MatMul_4:product:07create_message/dense_8/BiasAdd_4/ReadVariableOp:value:0*
T0*
_output_shapes
:	@2"
 create_message/dense_8/BiasAdd_4
create_message/dense_8/Tanh_4Tanh)create_message/dense_8/BiasAdd_4:output:0*
T0*
_output_shapes
:	@2
create_message/dense_8/Tanh_4Ö
.create_message/dense_9/MatMul_4/ReadVariableOpReadVariableOp5create_message_dense_9_matmul_readvariableop_resource*
_output_shapes

:@*
dtype020
.create_message/dense_9/MatMul_4/ReadVariableOpŃ
create_message/dense_9/MatMul_4MatMul!create_message/dense_8/Tanh_4:y:06create_message/dense_9/MatMul_4/ReadVariableOp:value:0*
T0*
_output_shapes
:	2!
create_message/dense_9/MatMul_4Ő
/create_message/dense_9/BiasAdd_4/ReadVariableOpReadVariableOp6create_message_dense_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype021
/create_message/dense_9/BiasAdd_4/ReadVariableOpÝ
 create_message/dense_9/BiasAdd_4BiasAdd)create_message/dense_9/MatMul_4:product:07create_message/dense_9/BiasAdd_4/ReadVariableOp:value:0*
T0*
_output_shapes
:	2"
 create_message/dense_9/BiasAdd_4
create_message/dense_9/Tanh_4Tanh)create_message/dense_9/BiasAdd_4:output:0*
T0*
_output_shapes
:	2
create_message/dense_9/Tanh_4¸
PartitionedCall_4PartitionedCall!create_message/dense_9/Tanh_4:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:J * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *-
f(R&
$__inference_message_aggregation_12562
PartitionedCall_4`
concat_9/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat_9/axis§
concat_9ConcatV2link_update/dense_12/Tanh_3:y:0PartitionedCall_4:output:0concat_9/axis:output:0*
N*
T0*
_output_shapes

:J02

concat_9Ń
,link_update/dense_10/MatMul_4/ReadVariableOpReadVariableOp3link_update_dense_10_matmul_readvariableop_resource*
_output_shapes
:	0*
dtype02.
,link_update/dense_10/MatMul_4/ReadVariableOpť
link_update/dense_10/MatMul_4MatMulconcat_9:output:04link_update/dense_10/MatMul_4/ReadVariableOp:value:0*
T0*
_output_shapes
:	J2
link_update/dense_10/MatMul_4Đ
-link_update/dense_10/BiasAdd_4/ReadVariableOpReadVariableOp4link_update_dense_10_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02/
-link_update/dense_10/BiasAdd_4/ReadVariableOpŐ
link_update/dense_10/BiasAdd_4BiasAdd'link_update/dense_10/MatMul_4:product:05link_update/dense_10/BiasAdd_4/ReadVariableOp:value:0*
T0*
_output_shapes
:	J2 
link_update/dense_10/BiasAdd_4
link_update/dense_10/Tanh_4Tanh'link_update/dense_10/BiasAdd_4:output:0*
T0*
_output_shapes
:	J2
link_update/dense_10/Tanh_4Ń
,link_update/dense_11/MatMul_4/ReadVariableOpReadVariableOp3link_update_dense_11_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype02.
,link_update/dense_11/MatMul_4/ReadVariableOpČ
link_update/dense_11/MatMul_4MatMullink_update/dense_10/Tanh_4:y:04link_update/dense_11/MatMul_4/ReadVariableOp:value:0*
T0*
_output_shapes

:J@2
link_update/dense_11/MatMul_4Ď
-link_update/dense_11/BiasAdd_4/ReadVariableOpReadVariableOp4link_update_dense_11_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-link_update/dense_11/BiasAdd_4/ReadVariableOpÔ
link_update/dense_11/BiasAdd_4BiasAdd'link_update/dense_11/MatMul_4:product:05link_update/dense_11/BiasAdd_4/ReadVariableOp:value:0*
T0*
_output_shapes

:J@2 
link_update/dense_11/BiasAdd_4
link_update/dense_11/Tanh_4Tanh'link_update/dense_11/BiasAdd_4:output:0*
T0*
_output_shapes

:J@2
link_update/dense_11/Tanh_4Đ
,link_update/dense_12/MatMul_4/ReadVariableOpReadVariableOp3link_update_dense_12_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02.
,link_update/dense_12/MatMul_4/ReadVariableOpČ
link_update/dense_12/MatMul_4MatMullink_update/dense_11/Tanh_4:y:04link_update/dense_12/MatMul_4/ReadVariableOp:value:0*
T0*
_output_shapes

:J2
link_update/dense_12/MatMul_4Ď
-link_update/dense_12/BiasAdd_4/ReadVariableOpReadVariableOp4link_update_dense_12_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-link_update/dense_12/BiasAdd_4/ReadVariableOpÔ
link_update/dense_12/BiasAdd_4BiasAdd'link_update/dense_12/MatMul_4:product:05link_update/dense_12/BiasAdd_4/ReadVariableOp:value:0*
T0*
_output_shapes

:J2 
link_update/dense_12/BiasAdd_4
link_update/dense_12/Tanh_4Tanh'link_update/dense_12/BiasAdd_4:output:0*
T0*
_output_shapes

:J2
link_update/dense_12/Tanh_4	
GatherV2_10/indicesConst*
_output_shapes	
:*
dtype0*ą
value§B¤"                                                      
   
   
   
   
                                                                                                                                            !   !   !   !   !   !   "   "   "   "   "   "   "   "   #   #   #   %   %   %   %   %   %   &   &   &   '   '   '   '   '   '   '   '   (   (   (   (   )   )   )   )   )   )   )   )   *   *   +   +   +   +   +   +   ,   ,   ,   ,   -   -   -   -   -   -   -   -   .   .   .   .   /   /   /   0   0   0   0   0   1   1   1   1   1   1   1   1   2   2   2   2   3   3   3   3   4   4   4   4   4   5   5   5   6   6   6   6   7   7   7   8   8   8   8   8   8   9   9   9   9   9   9   9   9   :   :   :   :   ;   ;   ;   ;   <   <   =   =   =   =   >   >   >   >   ?   ?   ?   ?   ?   @   @   A   A   A   A   A   A   B   B   B   B   C   C   C   C   C   C   C   C   D   D   D   D   E   E   E   E   F   F   F   F   G   G   G   G   H   H   H   H   H   I   I   I   I   I   I   I   I   2
GatherV2_10/indicesf
GatherV2_10/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2_10/axisÍ
GatherV2_10GatherV2link_update/dense_12/Tanh_4:y:0GatherV2_10/indices:output:0GatherV2_10/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:	2
GatherV2_10	
GatherV2_11/indicesConst*
_output_shapes	
:*
dtype0*ą
value§B¤"      
                                  #   %   &   '   5   6   7   8   9          D   E   F   G   H   I                                                                         !   "                                 B   C   D   E   F   G   H   I   :   ;   <   =   >   ?   @   A         
   D   E   F   G   H   I   2   3   4   :   ;   <   =   >   ?   @   A   *   +   ,   -   :   ;   <   =   >   ?   @   A   (   )   D   E   F   G   H   I   .   /   0   1   :   ;   <   =   >   ?   @   A   *   +   ,   -   2   3   4   5   6   7   8   9   :   ;   <   =   >   ?   @   A   #   %   &   '   .   /   0   1   5   6   7   8   9         
   .   /   0   1   2   3   4   D   E   F   G   H   I   :   ;   <   =   >   ?   @   A          !   "   #   %   &   '   (   )   *   +   ,   -   .   /   0   1   5   6   7   8   9   B   C   D   E   F   G   H   I          !   "   :   ;   <   =   >   ?   @   A                      !   "   #   %   &   '   *   +   ,   -   5   6   7   8   9   :   ;   <   =   >   ?   @   A   2
GatherV2_11/indicesf
GatherV2_11/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2_11/axisÍ
GatherV2_11GatherV2link_update/dense_12/Tanh_4:y:0GatherV2_11/indices:output:0GatherV2_11/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:	2
GatherV2_11b
concat_10/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat_10/axis
	concat_10ConcatV2GatherV2_10:output:0GatherV2_11:output:0concat_10/axis:output:0*
N*
T0*
_output_shapes
:	 2
	concat_10Ö
.create_message/dense_8/MatMul_5/ReadVariableOpReadVariableOp5create_message_dense_8_matmul_readvariableop_resource*
_output_shapes

: @*
dtype020
.create_message/dense_8/MatMul_5/ReadVariableOpÂ
create_message/dense_8/MatMul_5MatMulconcat_10:output:06create_message/dense_8/MatMul_5/ReadVariableOp:value:0*
T0*
_output_shapes
:	@2!
create_message/dense_8/MatMul_5Ő
/create_message/dense_8/BiasAdd_5/ReadVariableOpReadVariableOp6create_message_dense_8_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype021
/create_message/dense_8/BiasAdd_5/ReadVariableOpÝ
 create_message/dense_8/BiasAdd_5BiasAdd)create_message/dense_8/MatMul_5:product:07create_message/dense_8/BiasAdd_5/ReadVariableOp:value:0*
T0*
_output_shapes
:	@2"
 create_message/dense_8/BiasAdd_5
create_message/dense_8/Tanh_5Tanh)create_message/dense_8/BiasAdd_5:output:0*
T0*
_output_shapes
:	@2
create_message/dense_8/Tanh_5Ö
.create_message/dense_9/MatMul_5/ReadVariableOpReadVariableOp5create_message_dense_9_matmul_readvariableop_resource*
_output_shapes

:@*
dtype020
.create_message/dense_9/MatMul_5/ReadVariableOpŃ
create_message/dense_9/MatMul_5MatMul!create_message/dense_8/Tanh_5:y:06create_message/dense_9/MatMul_5/ReadVariableOp:value:0*
T0*
_output_shapes
:	2!
create_message/dense_9/MatMul_5Ő
/create_message/dense_9/BiasAdd_5/ReadVariableOpReadVariableOp6create_message_dense_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype021
/create_message/dense_9/BiasAdd_5/ReadVariableOpÝ
 create_message/dense_9/BiasAdd_5BiasAdd)create_message/dense_9/MatMul_5:product:07create_message/dense_9/BiasAdd_5/ReadVariableOp:value:0*
T0*
_output_shapes
:	2"
 create_message/dense_9/BiasAdd_5
create_message/dense_9/Tanh_5Tanh)create_message/dense_9/BiasAdd_5:output:0*
T0*
_output_shapes
:	2
create_message/dense_9/Tanh_5¸
PartitionedCall_5PartitionedCall!create_message/dense_9/Tanh_5:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:J * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *-
f(R&
$__inference_message_aggregation_12562
PartitionedCall_5b
concat_11/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat_11/axisŞ
	concat_11ConcatV2link_update/dense_12/Tanh_4:y:0PartitionedCall_5:output:0concat_11/axis:output:0*
N*
T0*
_output_shapes

:J02
	concat_11Ń
,link_update/dense_10/MatMul_5/ReadVariableOpReadVariableOp3link_update_dense_10_matmul_readvariableop_resource*
_output_shapes
:	0*
dtype02.
,link_update/dense_10/MatMul_5/ReadVariableOpź
link_update/dense_10/MatMul_5MatMulconcat_11:output:04link_update/dense_10/MatMul_5/ReadVariableOp:value:0*
T0*
_output_shapes
:	J2
link_update/dense_10/MatMul_5Đ
-link_update/dense_10/BiasAdd_5/ReadVariableOpReadVariableOp4link_update_dense_10_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02/
-link_update/dense_10/BiasAdd_5/ReadVariableOpŐ
link_update/dense_10/BiasAdd_5BiasAdd'link_update/dense_10/MatMul_5:product:05link_update/dense_10/BiasAdd_5/ReadVariableOp:value:0*
T0*
_output_shapes
:	J2 
link_update/dense_10/BiasAdd_5
link_update/dense_10/Tanh_5Tanh'link_update/dense_10/BiasAdd_5:output:0*
T0*
_output_shapes
:	J2
link_update/dense_10/Tanh_5Ń
,link_update/dense_11/MatMul_5/ReadVariableOpReadVariableOp3link_update_dense_11_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype02.
,link_update/dense_11/MatMul_5/ReadVariableOpČ
link_update/dense_11/MatMul_5MatMullink_update/dense_10/Tanh_5:y:04link_update/dense_11/MatMul_5/ReadVariableOp:value:0*
T0*
_output_shapes

:J@2
link_update/dense_11/MatMul_5Ď
-link_update/dense_11/BiasAdd_5/ReadVariableOpReadVariableOp4link_update_dense_11_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-link_update/dense_11/BiasAdd_5/ReadVariableOpÔ
link_update/dense_11/BiasAdd_5BiasAdd'link_update/dense_11/MatMul_5:product:05link_update/dense_11/BiasAdd_5/ReadVariableOp:value:0*
T0*
_output_shapes

:J@2 
link_update/dense_11/BiasAdd_5
link_update/dense_11/Tanh_5Tanh'link_update/dense_11/BiasAdd_5:output:0*
T0*
_output_shapes

:J@2
link_update/dense_11/Tanh_5Đ
,link_update/dense_12/MatMul_5/ReadVariableOpReadVariableOp3link_update_dense_12_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02.
,link_update/dense_12/MatMul_5/ReadVariableOpČ
link_update/dense_12/MatMul_5MatMullink_update/dense_11/Tanh_5:y:04link_update/dense_12/MatMul_5/ReadVariableOp:value:0*
T0*
_output_shapes

:J2
link_update/dense_12/MatMul_5Ď
-link_update/dense_12/BiasAdd_5/ReadVariableOpReadVariableOp4link_update_dense_12_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-link_update/dense_12/BiasAdd_5/ReadVariableOpÔ
link_update/dense_12/BiasAdd_5BiasAdd'link_update/dense_12/MatMul_5:product:05link_update/dense_12/BiasAdd_5/ReadVariableOp:value:0*
T0*
_output_shapes

:J2 
link_update/dense_12/BiasAdd_5
link_update/dense_12/Tanh_5Tanh'link_update/dense_12/BiasAdd_5:output:0*
T0*
_output_shapes

:J2
link_update/dense_12/Tanh_5	
GatherV2_12/indicesConst*
_output_shapes	
:*
dtype0*ą
value§B¤"                                                      
   
   
   
   
                                                                                                                                            !   !   !   !   !   !   "   "   "   "   "   "   "   "   #   #   #   %   %   %   %   %   %   &   &   &   '   '   '   '   '   '   '   '   (   (   (   (   )   )   )   )   )   )   )   )   *   *   +   +   +   +   +   +   ,   ,   ,   ,   -   -   -   -   -   -   -   -   .   .   .   .   /   /   /   0   0   0   0   0   1   1   1   1   1   1   1   1   2   2   2   2   3   3   3   3   4   4   4   4   4   5   5   5   6   6   6   6   7   7   7   8   8   8   8   8   8   9   9   9   9   9   9   9   9   :   :   :   :   ;   ;   ;   ;   <   <   =   =   =   =   >   >   >   >   ?   ?   ?   ?   ?   @   @   A   A   A   A   A   A   B   B   B   B   C   C   C   C   C   C   C   C   D   D   D   D   E   E   E   E   F   F   F   F   G   G   G   G   H   H   H   H   H   I   I   I   I   I   I   I   I   2
GatherV2_12/indicesf
GatherV2_12/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2_12/axisÍ
GatherV2_12GatherV2link_update/dense_12/Tanh_5:y:0GatherV2_12/indices:output:0GatherV2_12/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:	2
GatherV2_12	
GatherV2_13/indicesConst*
_output_shapes	
:*
dtype0*ą
value§B¤"      
                                  #   %   &   '   5   6   7   8   9          D   E   F   G   H   I                                                                         !   "                                 B   C   D   E   F   G   H   I   :   ;   <   =   >   ?   @   A         
   D   E   F   G   H   I   2   3   4   :   ;   <   =   >   ?   @   A   *   +   ,   -   :   ;   <   =   >   ?   @   A   (   )   D   E   F   G   H   I   .   /   0   1   :   ;   <   =   >   ?   @   A   *   +   ,   -   2   3   4   5   6   7   8   9   :   ;   <   =   >   ?   @   A   #   %   &   '   .   /   0   1   5   6   7   8   9         
   .   /   0   1   2   3   4   D   E   F   G   H   I   :   ;   <   =   >   ?   @   A          !   "   #   %   &   '   (   )   *   +   ,   -   .   /   0   1   5   6   7   8   9   B   C   D   E   F   G   H   I          !   "   :   ;   <   =   >   ?   @   A                      !   "   #   %   &   '   *   +   ,   -   5   6   7   8   9   :   ;   <   =   >   ?   @   A   2
GatherV2_13/indicesf
GatherV2_13/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2_13/axisÍ
GatherV2_13GatherV2link_update/dense_12/Tanh_5:y:0GatherV2_13/indices:output:0GatherV2_13/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:	2
GatherV2_13b
concat_12/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat_12/axis
	concat_12ConcatV2GatherV2_12:output:0GatherV2_13:output:0concat_12/axis:output:0*
N*
T0*
_output_shapes
:	 2
	concat_12Ö
.create_message/dense_8/MatMul_6/ReadVariableOpReadVariableOp5create_message_dense_8_matmul_readvariableop_resource*
_output_shapes

: @*
dtype020
.create_message/dense_8/MatMul_6/ReadVariableOpÂ
create_message/dense_8/MatMul_6MatMulconcat_12:output:06create_message/dense_8/MatMul_6/ReadVariableOp:value:0*
T0*
_output_shapes
:	@2!
create_message/dense_8/MatMul_6Ő
/create_message/dense_8/BiasAdd_6/ReadVariableOpReadVariableOp6create_message_dense_8_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype021
/create_message/dense_8/BiasAdd_6/ReadVariableOpÝ
 create_message/dense_8/BiasAdd_6BiasAdd)create_message/dense_8/MatMul_6:product:07create_message/dense_8/BiasAdd_6/ReadVariableOp:value:0*
T0*
_output_shapes
:	@2"
 create_message/dense_8/BiasAdd_6
create_message/dense_8/Tanh_6Tanh)create_message/dense_8/BiasAdd_6:output:0*
T0*
_output_shapes
:	@2
create_message/dense_8/Tanh_6Ö
.create_message/dense_9/MatMul_6/ReadVariableOpReadVariableOp5create_message_dense_9_matmul_readvariableop_resource*
_output_shapes

:@*
dtype020
.create_message/dense_9/MatMul_6/ReadVariableOpŃ
create_message/dense_9/MatMul_6MatMul!create_message/dense_8/Tanh_6:y:06create_message/dense_9/MatMul_6/ReadVariableOp:value:0*
T0*
_output_shapes
:	2!
create_message/dense_9/MatMul_6Ő
/create_message/dense_9/BiasAdd_6/ReadVariableOpReadVariableOp6create_message_dense_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype021
/create_message/dense_9/BiasAdd_6/ReadVariableOpÝ
 create_message/dense_9/BiasAdd_6BiasAdd)create_message/dense_9/MatMul_6:product:07create_message/dense_9/BiasAdd_6/ReadVariableOp:value:0*
T0*
_output_shapes
:	2"
 create_message/dense_9/BiasAdd_6
create_message/dense_9/Tanh_6Tanh)create_message/dense_9/BiasAdd_6:output:0*
T0*
_output_shapes
:	2
create_message/dense_9/Tanh_6¸
PartitionedCall_6PartitionedCall!create_message/dense_9/Tanh_6:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:J * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *-
f(R&
$__inference_message_aggregation_12562
PartitionedCall_6b
concat_13/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat_13/axisŞ
	concat_13ConcatV2link_update/dense_12/Tanh_5:y:0PartitionedCall_6:output:0concat_13/axis:output:0*
N*
T0*
_output_shapes

:J02
	concat_13Ń
,link_update/dense_10/MatMul_6/ReadVariableOpReadVariableOp3link_update_dense_10_matmul_readvariableop_resource*
_output_shapes
:	0*
dtype02.
,link_update/dense_10/MatMul_6/ReadVariableOpź
link_update/dense_10/MatMul_6MatMulconcat_13:output:04link_update/dense_10/MatMul_6/ReadVariableOp:value:0*
T0*
_output_shapes
:	J2
link_update/dense_10/MatMul_6Đ
-link_update/dense_10/BiasAdd_6/ReadVariableOpReadVariableOp4link_update_dense_10_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02/
-link_update/dense_10/BiasAdd_6/ReadVariableOpŐ
link_update/dense_10/BiasAdd_6BiasAdd'link_update/dense_10/MatMul_6:product:05link_update/dense_10/BiasAdd_6/ReadVariableOp:value:0*
T0*
_output_shapes
:	J2 
link_update/dense_10/BiasAdd_6
link_update/dense_10/Tanh_6Tanh'link_update/dense_10/BiasAdd_6:output:0*
T0*
_output_shapes
:	J2
link_update/dense_10/Tanh_6Ń
,link_update/dense_11/MatMul_6/ReadVariableOpReadVariableOp3link_update_dense_11_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype02.
,link_update/dense_11/MatMul_6/ReadVariableOpČ
link_update/dense_11/MatMul_6MatMullink_update/dense_10/Tanh_6:y:04link_update/dense_11/MatMul_6/ReadVariableOp:value:0*
T0*
_output_shapes

:J@2
link_update/dense_11/MatMul_6Ď
-link_update/dense_11/BiasAdd_6/ReadVariableOpReadVariableOp4link_update_dense_11_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-link_update/dense_11/BiasAdd_6/ReadVariableOpÔ
link_update/dense_11/BiasAdd_6BiasAdd'link_update/dense_11/MatMul_6:product:05link_update/dense_11/BiasAdd_6/ReadVariableOp:value:0*
T0*
_output_shapes

:J@2 
link_update/dense_11/BiasAdd_6
link_update/dense_11/Tanh_6Tanh'link_update/dense_11/BiasAdd_6:output:0*
T0*
_output_shapes

:J@2
link_update/dense_11/Tanh_6Đ
,link_update/dense_12/MatMul_6/ReadVariableOpReadVariableOp3link_update_dense_12_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02.
,link_update/dense_12/MatMul_6/ReadVariableOpČ
link_update/dense_12/MatMul_6MatMullink_update/dense_11/Tanh_6:y:04link_update/dense_12/MatMul_6/ReadVariableOp:value:0*
T0*
_output_shapes

:J2
link_update/dense_12/MatMul_6Ď
-link_update/dense_12/BiasAdd_6/ReadVariableOpReadVariableOp4link_update_dense_12_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-link_update/dense_12/BiasAdd_6/ReadVariableOpÔ
link_update/dense_12/BiasAdd_6BiasAdd'link_update/dense_12/MatMul_6:product:05link_update/dense_12/BiasAdd_6/ReadVariableOp:value:0*
T0*
_output_shapes

:J2 
link_update/dense_12/BiasAdd_6
link_update/dense_12/Tanh_6Tanh'link_update/dense_12/BiasAdd_6:output:0*
T0*
_output_shapes

:J2
link_update/dense_12/Tanh_6	
GatherV2_14/indicesConst*
_output_shapes	
:*
dtype0*ą
value§B¤"                                                      
   
   
   
   
                                                                                                                                            !   !   !   !   !   !   "   "   "   "   "   "   "   "   #   #   #   %   %   %   %   %   %   &   &   &   '   '   '   '   '   '   '   '   (   (   (   (   )   )   )   )   )   )   )   )   *   *   +   +   +   +   +   +   ,   ,   ,   ,   -   -   -   -   -   -   -   -   .   .   .   .   /   /   /   0   0   0   0   0   1   1   1   1   1   1   1   1   2   2   2   2   3   3   3   3   4   4   4   4   4   5   5   5   6   6   6   6   7   7   7   8   8   8   8   8   8   9   9   9   9   9   9   9   9   :   :   :   :   ;   ;   ;   ;   <   <   =   =   =   =   >   >   >   >   ?   ?   ?   ?   ?   @   @   A   A   A   A   A   A   B   B   B   B   C   C   C   C   C   C   C   C   D   D   D   D   E   E   E   E   F   F   F   F   G   G   G   G   H   H   H   H   H   I   I   I   I   I   I   I   I   2
GatherV2_14/indicesf
GatherV2_14/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2_14/axisÍ
GatherV2_14GatherV2link_update/dense_12/Tanh_6:y:0GatherV2_14/indices:output:0GatherV2_14/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:	2
GatherV2_14	
GatherV2_15/indicesConst*
_output_shapes	
:*
dtype0*ą
value§B¤"      
                                  #   %   &   '   5   6   7   8   9          D   E   F   G   H   I                                                                         !   "                                 B   C   D   E   F   G   H   I   :   ;   <   =   >   ?   @   A         
   D   E   F   G   H   I   2   3   4   :   ;   <   =   >   ?   @   A   *   +   ,   -   :   ;   <   =   >   ?   @   A   (   )   D   E   F   G   H   I   .   /   0   1   :   ;   <   =   >   ?   @   A   *   +   ,   -   2   3   4   5   6   7   8   9   :   ;   <   =   >   ?   @   A   #   %   &   '   .   /   0   1   5   6   7   8   9         
   .   /   0   1   2   3   4   D   E   F   G   H   I   :   ;   <   =   >   ?   @   A          !   "   #   %   &   '   (   )   *   +   ,   -   .   /   0   1   5   6   7   8   9   B   C   D   E   F   G   H   I          !   "   :   ;   <   =   >   ?   @   A                      !   "   #   %   &   '   *   +   ,   -   5   6   7   8   9   :   ;   <   =   >   ?   @   A   2
GatherV2_15/indicesf
GatherV2_15/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2_15/axisÍ
GatherV2_15GatherV2link_update/dense_12/Tanh_6:y:0GatherV2_15/indices:output:0GatherV2_15/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:	2
GatherV2_15b
concat_14/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat_14/axis
	concat_14ConcatV2GatherV2_14:output:0GatherV2_15:output:0concat_14/axis:output:0*
N*
T0*
_output_shapes
:	 2
	concat_14Ö
.create_message/dense_8/MatMul_7/ReadVariableOpReadVariableOp5create_message_dense_8_matmul_readvariableop_resource*
_output_shapes

: @*
dtype020
.create_message/dense_8/MatMul_7/ReadVariableOpÂ
create_message/dense_8/MatMul_7MatMulconcat_14:output:06create_message/dense_8/MatMul_7/ReadVariableOp:value:0*
T0*
_output_shapes
:	@2!
create_message/dense_8/MatMul_7Ő
/create_message/dense_8/BiasAdd_7/ReadVariableOpReadVariableOp6create_message_dense_8_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype021
/create_message/dense_8/BiasAdd_7/ReadVariableOpÝ
 create_message/dense_8/BiasAdd_7BiasAdd)create_message/dense_8/MatMul_7:product:07create_message/dense_8/BiasAdd_7/ReadVariableOp:value:0*
T0*
_output_shapes
:	@2"
 create_message/dense_8/BiasAdd_7
create_message/dense_8/Tanh_7Tanh)create_message/dense_8/BiasAdd_7:output:0*
T0*
_output_shapes
:	@2
create_message/dense_8/Tanh_7Ö
.create_message/dense_9/MatMul_7/ReadVariableOpReadVariableOp5create_message_dense_9_matmul_readvariableop_resource*
_output_shapes

:@*
dtype020
.create_message/dense_9/MatMul_7/ReadVariableOpŃ
create_message/dense_9/MatMul_7MatMul!create_message/dense_8/Tanh_7:y:06create_message/dense_9/MatMul_7/ReadVariableOp:value:0*
T0*
_output_shapes
:	2!
create_message/dense_9/MatMul_7Ő
/create_message/dense_9/BiasAdd_7/ReadVariableOpReadVariableOp6create_message_dense_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype021
/create_message/dense_9/BiasAdd_7/ReadVariableOpÝ
 create_message/dense_9/BiasAdd_7BiasAdd)create_message/dense_9/MatMul_7:product:07create_message/dense_9/BiasAdd_7/ReadVariableOp:value:0*
T0*
_output_shapes
:	2"
 create_message/dense_9/BiasAdd_7
create_message/dense_9/Tanh_7Tanh)create_message/dense_9/BiasAdd_7:output:0*
T0*
_output_shapes
:	2
create_message/dense_9/Tanh_7¸
PartitionedCall_7PartitionedCall!create_message/dense_9/Tanh_7:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:J * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *-
f(R&
$__inference_message_aggregation_12562
PartitionedCall_7b
concat_15/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat_15/axisŞ
	concat_15ConcatV2link_update/dense_12/Tanh_6:y:0PartitionedCall_7:output:0concat_15/axis:output:0*
N*
T0*
_output_shapes

:J02
	concat_15Ń
,link_update/dense_10/MatMul_7/ReadVariableOpReadVariableOp3link_update_dense_10_matmul_readvariableop_resource*
_output_shapes
:	0*
dtype02.
,link_update/dense_10/MatMul_7/ReadVariableOpź
link_update/dense_10/MatMul_7MatMulconcat_15:output:04link_update/dense_10/MatMul_7/ReadVariableOp:value:0*
T0*
_output_shapes
:	J2
link_update/dense_10/MatMul_7Đ
-link_update/dense_10/BiasAdd_7/ReadVariableOpReadVariableOp4link_update_dense_10_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02/
-link_update/dense_10/BiasAdd_7/ReadVariableOpŐ
link_update/dense_10/BiasAdd_7BiasAdd'link_update/dense_10/MatMul_7:product:05link_update/dense_10/BiasAdd_7/ReadVariableOp:value:0*
T0*
_output_shapes
:	J2 
link_update/dense_10/BiasAdd_7
link_update/dense_10/Tanh_7Tanh'link_update/dense_10/BiasAdd_7:output:0*
T0*
_output_shapes
:	J2
link_update/dense_10/Tanh_7Ń
,link_update/dense_11/MatMul_7/ReadVariableOpReadVariableOp3link_update_dense_11_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype02.
,link_update/dense_11/MatMul_7/ReadVariableOpČ
link_update/dense_11/MatMul_7MatMullink_update/dense_10/Tanh_7:y:04link_update/dense_11/MatMul_7/ReadVariableOp:value:0*
T0*
_output_shapes

:J@2
link_update/dense_11/MatMul_7Ď
-link_update/dense_11/BiasAdd_7/ReadVariableOpReadVariableOp4link_update_dense_11_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-link_update/dense_11/BiasAdd_7/ReadVariableOpÔ
link_update/dense_11/BiasAdd_7BiasAdd'link_update/dense_11/MatMul_7:product:05link_update/dense_11/BiasAdd_7/ReadVariableOp:value:0*
T0*
_output_shapes

:J@2 
link_update/dense_11/BiasAdd_7
link_update/dense_11/Tanh_7Tanh'link_update/dense_11/BiasAdd_7:output:0*
T0*
_output_shapes

:J@2
link_update/dense_11/Tanh_7Đ
,link_update/dense_12/MatMul_7/ReadVariableOpReadVariableOp3link_update_dense_12_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02.
,link_update/dense_12/MatMul_7/ReadVariableOpČ
link_update/dense_12/MatMul_7MatMullink_update/dense_11/Tanh_7:y:04link_update/dense_12/MatMul_7/ReadVariableOp:value:0*
T0*
_output_shapes

:J2
link_update/dense_12/MatMul_7Ď
-link_update/dense_12/BiasAdd_7/ReadVariableOpReadVariableOp4link_update_dense_12_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-link_update/dense_12/BiasAdd_7/ReadVariableOpÔ
link_update/dense_12/BiasAdd_7BiasAdd'link_update/dense_12/MatMul_7:product:05link_update/dense_12/BiasAdd_7/ReadVariableOp:value:0*
T0*
_output_shapes

:J2 
link_update/dense_12/BiasAdd_7
link_update/dense_12/Tanh_7Tanh'link_update/dense_12/BiasAdd_7:output:0*
T0*
_output_shapes

:J2
link_update/dense_12/Tanh_7j
IdentityIdentitylink_update/dense_12/Tanh_7:y:0*
T0*
_output_shapes

:J2

Identity"
identityIdentity:output:0*B
_input_shapes1
/::::::::::::B >

_output_shapes	
:

_user_specified_nameinput
Ăł

 __inference_message_passing_1534	
input9
5create_message_dense_8_matmul_readvariableop_resource:
6create_message_dense_8_biasadd_readvariableop_resource9
5create_message_dense_9_matmul_readvariableop_resource:
6create_message_dense_9_biasadd_readvariableop_resource7
3link_update_dense_10_matmul_readvariableop_resource8
4link_update_dense_10_biasadd_readvariableop_resource7
3link_update_dense_11_matmul_readvariableop_resource8
4link_update_dense_11_biasadd_readvariableop_resource7
3link_update_dense_12_matmul_readvariableop_resource8
4link_update_dense_12_biasadd_readvariableop_resource
identityo
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"   J   2
Reshape/shapee
ReshapeReshapeinputReshape/shape:output:0*
T0*
_output_shapes

:J2	
Reshapeq
transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose/permw
	transpose	TransposeReshape:output:0transpose/perm:output:0*
T0*
_output_shapes

:J2
	transpose}
Pad/paddingsConst*
_output_shapes

:*
dtype0*)
value B"               2
Pad/paddings`
PadPadtranspose:y:0Pad/paddings:output:0*
T0*
_output_shapes

:J2
Pad	
GatherV2/indicesConst*
_output_shapes	
:*
dtype0*ą
value§B¤"                                                      
   
   
   
   
                                                                                                                                            !   !   !   !   !   !   "   "   "   "   "   "   "   "   #   #   #   %   %   %   %   %   %   &   &   &   '   '   '   '   '   '   '   '   (   (   (   (   )   )   )   )   )   )   )   )   *   *   +   +   +   +   +   +   ,   ,   ,   ,   -   -   -   -   -   -   -   -   .   .   .   .   /   /   /   0   0   0   0   0   1   1   1   1   1   1   1   1   2   2   2   2   3   3   3   3   4   4   4   4   4   5   5   5   6   6   6   6   7   7   7   8   8   8   8   8   8   9   9   9   9   9   9   9   9   :   :   :   :   ;   ;   ;   ;   <   <   =   =   =   =   >   >   >   >   ?   ?   ?   ?   ?   @   @   A   A   A   A   A   A   B   B   B   B   C   C   C   C   C   C   C   C   D   D   D   D   E   E   E   E   F   F   F   F   G   G   G   G   H   H   H   H   H   I   I   I   I   I   I   I   I   2
GatherV2/indices`
GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2/axisŽ
GatherV2GatherV2Pad:output:0GatherV2/indices:output:0GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:	2

GatherV2	
GatherV2_1/indicesConst*
_output_shapes	
:*
dtype0*ą
value§B¤"      
                                  #   %   &   '   5   6   7   8   9          D   E   F   G   H   I                                                                         !   "                                 B   C   D   E   F   G   H   I   :   ;   <   =   >   ?   @   A         
   D   E   F   G   H   I   2   3   4   :   ;   <   =   >   ?   @   A   *   +   ,   -   :   ;   <   =   >   ?   @   A   (   )   D   E   F   G   H   I   .   /   0   1   :   ;   <   =   >   ?   @   A   *   +   ,   -   2   3   4   5   6   7   8   9   :   ;   <   =   >   ?   @   A   #   %   &   '   .   /   0   1   5   6   7   8   9         
   .   /   0   1   2   3   4   D   E   F   G   H   I   :   ;   <   =   >   ?   @   A          !   "   #   %   &   '   (   )   *   +   ,   -   .   /   0   1   5   6   7   8   9   B   C   D   E   F   G   H   I          !   "   :   ;   <   =   >   ?   @   A                      !   "   #   %   &   '   *   +   ,   -   5   6   7   8   9   :   ;   <   =   >   ?   @   A   2
GatherV2_1/indicesd
GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2_1/axisś

GatherV2_1GatherV2Pad:output:0GatherV2_1/indices:output:0GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:	2

GatherV2_1\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis
concatConcatV2GatherV2:output:0GatherV2_1:output:0concat/axis:output:0*
N*
T0*
_output_shapes
:	 2
concatŇ
,create_message/dense_8/MatMul/ReadVariableOpReadVariableOp5create_message_dense_8_matmul_readvariableop_resource*
_output_shapes

: @*
dtype02.
,create_message/dense_8/MatMul/ReadVariableOpš
create_message/dense_8/MatMulMatMulconcat:output:04create_message/dense_8/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	@2
create_message/dense_8/MatMulŃ
-create_message/dense_8/BiasAdd/ReadVariableOpReadVariableOp6create_message_dense_8_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-create_message/dense_8/BiasAdd/ReadVariableOpŐ
create_message/dense_8/BiasAddBiasAdd'create_message/dense_8/MatMul:product:05create_message/dense_8/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	@2 
create_message/dense_8/BiasAdd
create_message/dense_8/TanhTanh'create_message/dense_8/BiasAdd:output:0*
T0*
_output_shapes
:	@2
create_message/dense_8/TanhŇ
,create_message/dense_9/MatMul/ReadVariableOpReadVariableOp5create_message_dense_9_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02.
,create_message/dense_9/MatMul/ReadVariableOpÉ
create_message/dense_9/MatMulMatMulcreate_message/dense_8/Tanh:y:04create_message/dense_9/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
create_message/dense_9/MatMulŃ
-create_message/dense_9/BiasAdd/ReadVariableOpReadVariableOp6create_message_dense_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-create_message/dense_9/BiasAdd/ReadVariableOpŐ
create_message/dense_9/BiasAddBiasAdd'create_message/dense_9/MatMul:product:05create_message/dense_9/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	2 
create_message/dense_9/BiasAdd
create_message/dense_9/TanhTanh'create_message/dense_9/BiasAdd:output:0*
T0*
_output_shapes
:	2
create_message/dense_9/Tanhĺ
PartitionedCallPartitionedCallcreate_message/dense_9/Tanh:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *Q
_output_shapes?
=:J : ::J:	: ::J: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *-
f(R&
$__inference_message_aggregation_12562
PartitionedCall`
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat_1/axis
concat_1ConcatV2Pad:output:0PartitionedCall:output:0concat_1/axis:output:0*
N*
T0*
_output_shapes

:J02

concat_1Í
*link_update/dense_10/MatMul/ReadVariableOpReadVariableOp3link_update_dense_10_matmul_readvariableop_resource*
_output_shapes
:	0*
dtype02,
*link_update/dense_10/MatMul/ReadVariableOpľ
link_update/dense_10/MatMulMatMulconcat_1:output:02link_update/dense_10/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	J2
link_update/dense_10/MatMulĚ
+link_update/dense_10/BiasAdd/ReadVariableOpReadVariableOp4link_update_dense_10_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02-
+link_update/dense_10/BiasAdd/ReadVariableOpÍ
link_update/dense_10/BiasAddBiasAdd%link_update/dense_10/MatMul:product:03link_update/dense_10/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	J2
link_update/dense_10/BiasAdd
link_update/dense_10/TanhTanh%link_update/dense_10/BiasAdd:output:0*
T0*
_output_shapes
:	J2
link_update/dense_10/TanhÍ
*link_update/dense_11/MatMul/ReadVariableOpReadVariableOp3link_update_dense_11_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype02,
*link_update/dense_11/MatMul/ReadVariableOpŔ
link_update/dense_11/MatMulMatMullink_update/dense_10/Tanh:y:02link_update/dense_11/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:J@2
link_update/dense_11/MatMulË
+link_update/dense_11/BiasAdd/ReadVariableOpReadVariableOp4link_update_dense_11_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02-
+link_update/dense_11/BiasAdd/ReadVariableOpĚ
link_update/dense_11/BiasAddBiasAdd%link_update/dense_11/MatMul:product:03link_update/dense_11/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:J@2
link_update/dense_11/BiasAdd
link_update/dense_11/TanhTanh%link_update/dense_11/BiasAdd:output:0*
T0*
_output_shapes

:J@2
link_update/dense_11/TanhĚ
*link_update/dense_12/MatMul/ReadVariableOpReadVariableOp3link_update_dense_12_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02,
*link_update/dense_12/MatMul/ReadVariableOpŔ
link_update/dense_12/MatMulMatMullink_update/dense_11/Tanh:y:02link_update/dense_12/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:J2
link_update/dense_12/MatMulË
+link_update/dense_12/BiasAdd/ReadVariableOpReadVariableOp4link_update_dense_12_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02-
+link_update/dense_12/BiasAdd/ReadVariableOpĚ
link_update/dense_12/BiasAddBiasAdd%link_update/dense_12/MatMul:product:03link_update/dense_12/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:J2
link_update/dense_12/BiasAdd
link_update/dense_12/TanhTanh%link_update/dense_12/BiasAdd:output:0*
T0*
_output_shapes

:J2
link_update/dense_12/Tanh	
GatherV2_2/indicesConst*
_output_shapes	
:*
dtype0*ą
value§B¤"                                                      
   
   
   
   
                                                                                                                                            !   !   !   !   !   !   "   "   "   "   "   "   "   "   #   #   #   %   %   %   %   %   %   &   &   &   '   '   '   '   '   '   '   '   (   (   (   (   )   )   )   )   )   )   )   )   *   *   +   +   +   +   +   +   ,   ,   ,   ,   -   -   -   -   -   -   -   -   .   .   .   .   /   /   /   0   0   0   0   0   1   1   1   1   1   1   1   1   2   2   2   2   3   3   3   3   4   4   4   4   4   5   5   5   6   6   6   6   7   7   7   8   8   8   8   8   8   9   9   9   9   9   9   9   9   :   :   :   :   ;   ;   ;   ;   <   <   =   =   =   =   >   >   >   >   ?   ?   ?   ?   ?   @   @   A   A   A   A   A   A   B   B   B   B   C   C   C   C   C   C   C   C   D   D   D   D   E   E   E   E   F   F   F   F   G   G   G   G   H   H   H   H   H   I   I   I   I   I   I   I   I   2
GatherV2_2/indicesd
GatherV2_2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2_2/axisÇ

GatherV2_2GatherV2link_update/dense_12/Tanh:y:0GatherV2_2/indices:output:0GatherV2_2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:	2

GatherV2_2	
GatherV2_3/indicesConst*
_output_shapes	
:*
dtype0*ą
value§B¤"      
                                  #   %   &   '   5   6   7   8   9          D   E   F   G   H   I                                                                         !   "                                 B   C   D   E   F   G   H   I   :   ;   <   =   >   ?   @   A         
   D   E   F   G   H   I   2   3   4   :   ;   <   =   >   ?   @   A   *   +   ,   -   :   ;   <   =   >   ?   @   A   (   )   D   E   F   G   H   I   .   /   0   1   :   ;   <   =   >   ?   @   A   *   +   ,   -   2   3   4   5   6   7   8   9   :   ;   <   =   >   ?   @   A   #   %   &   '   .   /   0   1   5   6   7   8   9         
   .   /   0   1   2   3   4   D   E   F   G   H   I   :   ;   <   =   >   ?   @   A          !   "   #   %   &   '   (   )   *   +   ,   -   .   /   0   1   5   6   7   8   9   B   C   D   E   F   G   H   I          !   "   :   ;   <   =   >   ?   @   A                      !   "   #   %   &   '   *   +   ,   -   5   6   7   8   9   :   ;   <   =   >   ?   @   A   2
GatherV2_3/indicesd
GatherV2_3/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2_3/axisÇ

GatherV2_3GatherV2link_update/dense_12/Tanh:y:0GatherV2_3/indices:output:0GatherV2_3/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:	2

GatherV2_3`
concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat_2/axis
concat_2ConcatV2GatherV2_2:output:0GatherV2_3:output:0concat_2/axis:output:0*
N*
T0*
_output_shapes
:	 2

concat_2Ö
.create_message/dense_8/MatMul_1/ReadVariableOpReadVariableOp5create_message_dense_8_matmul_readvariableop_resource*
_output_shapes

: @*
dtype020
.create_message/dense_8/MatMul_1/ReadVariableOpÁ
create_message/dense_8/MatMul_1MatMulconcat_2:output:06create_message/dense_8/MatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	@2!
create_message/dense_8/MatMul_1Ő
/create_message/dense_8/BiasAdd_1/ReadVariableOpReadVariableOp6create_message_dense_8_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype021
/create_message/dense_8/BiasAdd_1/ReadVariableOpÝ
 create_message/dense_8/BiasAdd_1BiasAdd)create_message/dense_8/MatMul_1:product:07create_message/dense_8/BiasAdd_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	@2"
 create_message/dense_8/BiasAdd_1
create_message/dense_8/Tanh_1Tanh)create_message/dense_8/BiasAdd_1:output:0*
T0*
_output_shapes
:	@2
create_message/dense_8/Tanh_1Ö
.create_message/dense_9/MatMul_1/ReadVariableOpReadVariableOp5create_message_dense_9_matmul_readvariableop_resource*
_output_shapes

:@*
dtype020
.create_message/dense_9/MatMul_1/ReadVariableOpŃ
create_message/dense_9/MatMul_1MatMul!create_message/dense_8/Tanh_1:y:06create_message/dense_9/MatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	2!
create_message/dense_9/MatMul_1Ő
/create_message/dense_9/BiasAdd_1/ReadVariableOpReadVariableOp6create_message_dense_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype021
/create_message/dense_9/BiasAdd_1/ReadVariableOpÝ
 create_message/dense_9/BiasAdd_1BiasAdd)create_message/dense_9/MatMul_1:product:07create_message/dense_9/BiasAdd_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	2"
 create_message/dense_9/BiasAdd_1
create_message/dense_9/Tanh_1Tanh)create_message/dense_9/BiasAdd_1:output:0*
T0*
_output_shapes
:	2
create_message/dense_9/Tanh_1ë
PartitionedCall_1PartitionedCall!create_message/dense_9/Tanh_1:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *Q
_output_shapes?
=:J : ::J:	: ::J: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *-
f(R&
$__inference_message_aggregation_12562
PartitionedCall_1`
concat_3/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat_3/axisĽ
concat_3ConcatV2link_update/dense_12/Tanh:y:0PartitionedCall_1:output:0concat_3/axis:output:0*
N*
T0*
_output_shapes

:J02

concat_3Ń
,link_update/dense_10/MatMul_1/ReadVariableOpReadVariableOp3link_update_dense_10_matmul_readvariableop_resource*
_output_shapes
:	0*
dtype02.
,link_update/dense_10/MatMul_1/ReadVariableOpť
link_update/dense_10/MatMul_1MatMulconcat_3:output:04link_update/dense_10/MatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	J2
link_update/dense_10/MatMul_1Đ
-link_update/dense_10/BiasAdd_1/ReadVariableOpReadVariableOp4link_update_dense_10_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02/
-link_update/dense_10/BiasAdd_1/ReadVariableOpŐ
link_update/dense_10/BiasAdd_1BiasAdd'link_update/dense_10/MatMul_1:product:05link_update/dense_10/BiasAdd_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	J2 
link_update/dense_10/BiasAdd_1
link_update/dense_10/Tanh_1Tanh'link_update/dense_10/BiasAdd_1:output:0*
T0*
_output_shapes
:	J2
link_update/dense_10/Tanh_1Ń
,link_update/dense_11/MatMul_1/ReadVariableOpReadVariableOp3link_update_dense_11_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype02.
,link_update/dense_11/MatMul_1/ReadVariableOpČ
link_update/dense_11/MatMul_1MatMullink_update/dense_10/Tanh_1:y:04link_update/dense_11/MatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes

:J@2
link_update/dense_11/MatMul_1Ď
-link_update/dense_11/BiasAdd_1/ReadVariableOpReadVariableOp4link_update_dense_11_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-link_update/dense_11/BiasAdd_1/ReadVariableOpÔ
link_update/dense_11/BiasAdd_1BiasAdd'link_update/dense_11/MatMul_1:product:05link_update/dense_11/BiasAdd_1/ReadVariableOp:value:0*
T0*
_output_shapes

:J@2 
link_update/dense_11/BiasAdd_1
link_update/dense_11/Tanh_1Tanh'link_update/dense_11/BiasAdd_1:output:0*
T0*
_output_shapes

:J@2
link_update/dense_11/Tanh_1Đ
,link_update/dense_12/MatMul_1/ReadVariableOpReadVariableOp3link_update_dense_12_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02.
,link_update/dense_12/MatMul_1/ReadVariableOpČ
link_update/dense_12/MatMul_1MatMullink_update/dense_11/Tanh_1:y:04link_update/dense_12/MatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes

:J2
link_update/dense_12/MatMul_1Ď
-link_update/dense_12/BiasAdd_1/ReadVariableOpReadVariableOp4link_update_dense_12_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-link_update/dense_12/BiasAdd_1/ReadVariableOpÔ
link_update/dense_12/BiasAdd_1BiasAdd'link_update/dense_12/MatMul_1:product:05link_update/dense_12/BiasAdd_1/ReadVariableOp:value:0*
T0*
_output_shapes

:J2 
link_update/dense_12/BiasAdd_1
link_update/dense_12/Tanh_1Tanh'link_update/dense_12/BiasAdd_1:output:0*
T0*
_output_shapes

:J2
link_update/dense_12/Tanh_1	
GatherV2_4/indicesConst*
_output_shapes	
:*
dtype0*ą
value§B¤"                                                      
   
   
   
   
                                                                                                                                            !   !   !   !   !   !   "   "   "   "   "   "   "   "   #   #   #   %   %   %   %   %   %   &   &   &   '   '   '   '   '   '   '   '   (   (   (   (   )   )   )   )   )   )   )   )   *   *   +   +   +   +   +   +   ,   ,   ,   ,   -   -   -   -   -   -   -   -   .   .   .   .   /   /   /   0   0   0   0   0   1   1   1   1   1   1   1   1   2   2   2   2   3   3   3   3   4   4   4   4   4   5   5   5   6   6   6   6   7   7   7   8   8   8   8   8   8   9   9   9   9   9   9   9   9   :   :   :   :   ;   ;   ;   ;   <   <   =   =   =   =   >   >   >   >   ?   ?   ?   ?   ?   @   @   A   A   A   A   A   A   B   B   B   B   C   C   C   C   C   C   C   C   D   D   D   D   E   E   E   E   F   F   F   F   G   G   G   G   H   H   H   H   H   I   I   I   I   I   I   I   I   2
GatherV2_4/indicesd
GatherV2_4/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2_4/axisÉ

GatherV2_4GatherV2link_update/dense_12/Tanh_1:y:0GatherV2_4/indices:output:0GatherV2_4/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:	2

GatherV2_4	
GatherV2_5/indicesConst*
_output_shapes	
:*
dtype0*ą
value§B¤"      
                                  #   %   &   '   5   6   7   8   9          D   E   F   G   H   I                                                                         !   "                                 B   C   D   E   F   G   H   I   :   ;   <   =   >   ?   @   A         
   D   E   F   G   H   I   2   3   4   :   ;   <   =   >   ?   @   A   *   +   ,   -   :   ;   <   =   >   ?   @   A   (   )   D   E   F   G   H   I   .   /   0   1   :   ;   <   =   >   ?   @   A   *   +   ,   -   2   3   4   5   6   7   8   9   :   ;   <   =   >   ?   @   A   #   %   &   '   .   /   0   1   5   6   7   8   9         
   .   /   0   1   2   3   4   D   E   F   G   H   I   :   ;   <   =   >   ?   @   A          !   "   #   %   &   '   (   )   *   +   ,   -   .   /   0   1   5   6   7   8   9   B   C   D   E   F   G   H   I          !   "   :   ;   <   =   >   ?   @   A                      !   "   #   %   &   '   *   +   ,   -   5   6   7   8   9   :   ;   <   =   >   ?   @   A   2
GatherV2_5/indicesd
GatherV2_5/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2_5/axisÉ

GatherV2_5GatherV2link_update/dense_12/Tanh_1:y:0GatherV2_5/indices:output:0GatherV2_5/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:	2

GatherV2_5`
concat_4/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat_4/axis
concat_4ConcatV2GatherV2_4:output:0GatherV2_5:output:0concat_4/axis:output:0*
N*
T0*
_output_shapes
:	 2

concat_4Ö
.create_message/dense_8/MatMul_2/ReadVariableOpReadVariableOp5create_message_dense_8_matmul_readvariableop_resource*
_output_shapes

: @*
dtype020
.create_message/dense_8/MatMul_2/ReadVariableOpÁ
create_message/dense_8/MatMul_2MatMulconcat_4:output:06create_message/dense_8/MatMul_2/ReadVariableOp:value:0*
T0*
_output_shapes
:	@2!
create_message/dense_8/MatMul_2Ő
/create_message/dense_8/BiasAdd_2/ReadVariableOpReadVariableOp6create_message_dense_8_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype021
/create_message/dense_8/BiasAdd_2/ReadVariableOpÝ
 create_message/dense_8/BiasAdd_2BiasAdd)create_message/dense_8/MatMul_2:product:07create_message/dense_8/BiasAdd_2/ReadVariableOp:value:0*
T0*
_output_shapes
:	@2"
 create_message/dense_8/BiasAdd_2
create_message/dense_8/Tanh_2Tanh)create_message/dense_8/BiasAdd_2:output:0*
T0*
_output_shapes
:	@2
create_message/dense_8/Tanh_2Ö
.create_message/dense_9/MatMul_2/ReadVariableOpReadVariableOp5create_message_dense_9_matmul_readvariableop_resource*
_output_shapes

:@*
dtype020
.create_message/dense_9/MatMul_2/ReadVariableOpŃ
create_message/dense_9/MatMul_2MatMul!create_message/dense_8/Tanh_2:y:06create_message/dense_9/MatMul_2/ReadVariableOp:value:0*
T0*
_output_shapes
:	2!
create_message/dense_9/MatMul_2Ő
/create_message/dense_9/BiasAdd_2/ReadVariableOpReadVariableOp6create_message_dense_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype021
/create_message/dense_9/BiasAdd_2/ReadVariableOpÝ
 create_message/dense_9/BiasAdd_2BiasAdd)create_message/dense_9/MatMul_2:product:07create_message/dense_9/BiasAdd_2/ReadVariableOp:value:0*
T0*
_output_shapes
:	2"
 create_message/dense_9/BiasAdd_2
create_message/dense_9/Tanh_2Tanh)create_message/dense_9/BiasAdd_2:output:0*
T0*
_output_shapes
:	2
create_message/dense_9/Tanh_2ë
PartitionedCall_2PartitionedCall!create_message/dense_9/Tanh_2:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *Q
_output_shapes?
=:J : ::J:	: ::J: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *-
f(R&
$__inference_message_aggregation_12562
PartitionedCall_2`
concat_5/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat_5/axis§
concat_5ConcatV2link_update/dense_12/Tanh_1:y:0PartitionedCall_2:output:0concat_5/axis:output:0*
N*
T0*
_output_shapes

:J02

concat_5Ń
,link_update/dense_10/MatMul_2/ReadVariableOpReadVariableOp3link_update_dense_10_matmul_readvariableop_resource*
_output_shapes
:	0*
dtype02.
,link_update/dense_10/MatMul_2/ReadVariableOpť
link_update/dense_10/MatMul_2MatMulconcat_5:output:04link_update/dense_10/MatMul_2/ReadVariableOp:value:0*
T0*
_output_shapes
:	J2
link_update/dense_10/MatMul_2Đ
-link_update/dense_10/BiasAdd_2/ReadVariableOpReadVariableOp4link_update_dense_10_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02/
-link_update/dense_10/BiasAdd_2/ReadVariableOpŐ
link_update/dense_10/BiasAdd_2BiasAdd'link_update/dense_10/MatMul_2:product:05link_update/dense_10/BiasAdd_2/ReadVariableOp:value:0*
T0*
_output_shapes
:	J2 
link_update/dense_10/BiasAdd_2
link_update/dense_10/Tanh_2Tanh'link_update/dense_10/BiasAdd_2:output:0*
T0*
_output_shapes
:	J2
link_update/dense_10/Tanh_2Ń
,link_update/dense_11/MatMul_2/ReadVariableOpReadVariableOp3link_update_dense_11_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype02.
,link_update/dense_11/MatMul_2/ReadVariableOpČ
link_update/dense_11/MatMul_2MatMullink_update/dense_10/Tanh_2:y:04link_update/dense_11/MatMul_2/ReadVariableOp:value:0*
T0*
_output_shapes

:J@2
link_update/dense_11/MatMul_2Ď
-link_update/dense_11/BiasAdd_2/ReadVariableOpReadVariableOp4link_update_dense_11_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-link_update/dense_11/BiasAdd_2/ReadVariableOpÔ
link_update/dense_11/BiasAdd_2BiasAdd'link_update/dense_11/MatMul_2:product:05link_update/dense_11/BiasAdd_2/ReadVariableOp:value:0*
T0*
_output_shapes

:J@2 
link_update/dense_11/BiasAdd_2
link_update/dense_11/Tanh_2Tanh'link_update/dense_11/BiasAdd_2:output:0*
T0*
_output_shapes

:J@2
link_update/dense_11/Tanh_2Đ
,link_update/dense_12/MatMul_2/ReadVariableOpReadVariableOp3link_update_dense_12_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02.
,link_update/dense_12/MatMul_2/ReadVariableOpČ
link_update/dense_12/MatMul_2MatMullink_update/dense_11/Tanh_2:y:04link_update/dense_12/MatMul_2/ReadVariableOp:value:0*
T0*
_output_shapes

:J2
link_update/dense_12/MatMul_2Ď
-link_update/dense_12/BiasAdd_2/ReadVariableOpReadVariableOp4link_update_dense_12_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-link_update/dense_12/BiasAdd_2/ReadVariableOpÔ
link_update/dense_12/BiasAdd_2BiasAdd'link_update/dense_12/MatMul_2:product:05link_update/dense_12/BiasAdd_2/ReadVariableOp:value:0*
T0*
_output_shapes

:J2 
link_update/dense_12/BiasAdd_2
link_update/dense_12/Tanh_2Tanh'link_update/dense_12/BiasAdd_2:output:0*
T0*
_output_shapes

:J2
link_update/dense_12/Tanh_2	
GatherV2_6/indicesConst*
_output_shapes	
:*
dtype0*ą
value§B¤"                                                      
   
   
   
   
                                                                                                                                            !   !   !   !   !   !   "   "   "   "   "   "   "   "   #   #   #   %   %   %   %   %   %   &   &   &   '   '   '   '   '   '   '   '   (   (   (   (   )   )   )   )   )   )   )   )   *   *   +   +   +   +   +   +   ,   ,   ,   ,   -   -   -   -   -   -   -   -   .   .   .   .   /   /   /   0   0   0   0   0   1   1   1   1   1   1   1   1   2   2   2   2   3   3   3   3   4   4   4   4   4   5   5   5   6   6   6   6   7   7   7   8   8   8   8   8   8   9   9   9   9   9   9   9   9   :   :   :   :   ;   ;   ;   ;   <   <   =   =   =   =   >   >   >   >   ?   ?   ?   ?   ?   @   @   A   A   A   A   A   A   B   B   B   B   C   C   C   C   C   C   C   C   D   D   D   D   E   E   E   E   F   F   F   F   G   G   G   G   H   H   H   H   H   I   I   I   I   I   I   I   I   2
GatherV2_6/indicesd
GatherV2_6/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2_6/axisÉ

GatherV2_6GatherV2link_update/dense_12/Tanh_2:y:0GatherV2_6/indices:output:0GatherV2_6/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:	2

GatherV2_6	
GatherV2_7/indicesConst*
_output_shapes	
:*
dtype0*ą
value§B¤"      
                                  #   %   &   '   5   6   7   8   9          D   E   F   G   H   I                                                                         !   "                                 B   C   D   E   F   G   H   I   :   ;   <   =   >   ?   @   A         
   D   E   F   G   H   I   2   3   4   :   ;   <   =   >   ?   @   A   *   +   ,   -   :   ;   <   =   >   ?   @   A   (   )   D   E   F   G   H   I   .   /   0   1   :   ;   <   =   >   ?   @   A   *   +   ,   -   2   3   4   5   6   7   8   9   :   ;   <   =   >   ?   @   A   #   %   &   '   .   /   0   1   5   6   7   8   9         
   .   /   0   1   2   3   4   D   E   F   G   H   I   :   ;   <   =   >   ?   @   A          !   "   #   %   &   '   (   )   *   +   ,   -   .   /   0   1   5   6   7   8   9   B   C   D   E   F   G   H   I          !   "   :   ;   <   =   >   ?   @   A                      !   "   #   %   &   '   *   +   ,   -   5   6   7   8   9   :   ;   <   =   >   ?   @   A   2
GatherV2_7/indicesd
GatherV2_7/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2_7/axisÉ

GatherV2_7GatherV2link_update/dense_12/Tanh_2:y:0GatherV2_7/indices:output:0GatherV2_7/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:	2

GatherV2_7`
concat_6/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat_6/axis
concat_6ConcatV2GatherV2_6:output:0GatherV2_7:output:0concat_6/axis:output:0*
N*
T0*
_output_shapes
:	 2

concat_6Ö
.create_message/dense_8/MatMul_3/ReadVariableOpReadVariableOp5create_message_dense_8_matmul_readvariableop_resource*
_output_shapes

: @*
dtype020
.create_message/dense_8/MatMul_3/ReadVariableOpÁ
create_message/dense_8/MatMul_3MatMulconcat_6:output:06create_message/dense_8/MatMul_3/ReadVariableOp:value:0*
T0*
_output_shapes
:	@2!
create_message/dense_8/MatMul_3Ő
/create_message/dense_8/BiasAdd_3/ReadVariableOpReadVariableOp6create_message_dense_8_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype021
/create_message/dense_8/BiasAdd_3/ReadVariableOpÝ
 create_message/dense_8/BiasAdd_3BiasAdd)create_message/dense_8/MatMul_3:product:07create_message/dense_8/BiasAdd_3/ReadVariableOp:value:0*
T0*
_output_shapes
:	@2"
 create_message/dense_8/BiasAdd_3
create_message/dense_8/Tanh_3Tanh)create_message/dense_8/BiasAdd_3:output:0*
T0*
_output_shapes
:	@2
create_message/dense_8/Tanh_3Ö
.create_message/dense_9/MatMul_3/ReadVariableOpReadVariableOp5create_message_dense_9_matmul_readvariableop_resource*
_output_shapes

:@*
dtype020
.create_message/dense_9/MatMul_3/ReadVariableOpŃ
create_message/dense_9/MatMul_3MatMul!create_message/dense_8/Tanh_3:y:06create_message/dense_9/MatMul_3/ReadVariableOp:value:0*
T0*
_output_shapes
:	2!
create_message/dense_9/MatMul_3Ő
/create_message/dense_9/BiasAdd_3/ReadVariableOpReadVariableOp6create_message_dense_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype021
/create_message/dense_9/BiasAdd_3/ReadVariableOpÝ
 create_message/dense_9/BiasAdd_3BiasAdd)create_message/dense_9/MatMul_3:product:07create_message/dense_9/BiasAdd_3/ReadVariableOp:value:0*
T0*
_output_shapes
:	2"
 create_message/dense_9/BiasAdd_3
create_message/dense_9/Tanh_3Tanh)create_message/dense_9/BiasAdd_3:output:0*
T0*
_output_shapes
:	2
create_message/dense_9/Tanh_3ë
PartitionedCall_3PartitionedCall!create_message/dense_9/Tanh_3:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *Q
_output_shapes?
=:J : ::J:	: ::J: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *-
f(R&
$__inference_message_aggregation_12562
PartitionedCall_3`
concat_7/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat_7/axis§
concat_7ConcatV2link_update/dense_12/Tanh_2:y:0PartitionedCall_3:output:0concat_7/axis:output:0*
N*
T0*
_output_shapes

:J02

concat_7Ń
,link_update/dense_10/MatMul_3/ReadVariableOpReadVariableOp3link_update_dense_10_matmul_readvariableop_resource*
_output_shapes
:	0*
dtype02.
,link_update/dense_10/MatMul_3/ReadVariableOpť
link_update/dense_10/MatMul_3MatMulconcat_7:output:04link_update/dense_10/MatMul_3/ReadVariableOp:value:0*
T0*
_output_shapes
:	J2
link_update/dense_10/MatMul_3Đ
-link_update/dense_10/BiasAdd_3/ReadVariableOpReadVariableOp4link_update_dense_10_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02/
-link_update/dense_10/BiasAdd_3/ReadVariableOpŐ
link_update/dense_10/BiasAdd_3BiasAdd'link_update/dense_10/MatMul_3:product:05link_update/dense_10/BiasAdd_3/ReadVariableOp:value:0*
T0*
_output_shapes
:	J2 
link_update/dense_10/BiasAdd_3
link_update/dense_10/Tanh_3Tanh'link_update/dense_10/BiasAdd_3:output:0*
T0*
_output_shapes
:	J2
link_update/dense_10/Tanh_3Ń
,link_update/dense_11/MatMul_3/ReadVariableOpReadVariableOp3link_update_dense_11_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype02.
,link_update/dense_11/MatMul_3/ReadVariableOpČ
link_update/dense_11/MatMul_3MatMullink_update/dense_10/Tanh_3:y:04link_update/dense_11/MatMul_3/ReadVariableOp:value:0*
T0*
_output_shapes

:J@2
link_update/dense_11/MatMul_3Ď
-link_update/dense_11/BiasAdd_3/ReadVariableOpReadVariableOp4link_update_dense_11_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-link_update/dense_11/BiasAdd_3/ReadVariableOpÔ
link_update/dense_11/BiasAdd_3BiasAdd'link_update/dense_11/MatMul_3:product:05link_update/dense_11/BiasAdd_3/ReadVariableOp:value:0*
T0*
_output_shapes

:J@2 
link_update/dense_11/BiasAdd_3
link_update/dense_11/Tanh_3Tanh'link_update/dense_11/BiasAdd_3:output:0*
T0*
_output_shapes

:J@2
link_update/dense_11/Tanh_3Đ
,link_update/dense_12/MatMul_3/ReadVariableOpReadVariableOp3link_update_dense_12_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02.
,link_update/dense_12/MatMul_3/ReadVariableOpČ
link_update/dense_12/MatMul_3MatMullink_update/dense_11/Tanh_3:y:04link_update/dense_12/MatMul_3/ReadVariableOp:value:0*
T0*
_output_shapes

:J2
link_update/dense_12/MatMul_3Ď
-link_update/dense_12/BiasAdd_3/ReadVariableOpReadVariableOp4link_update_dense_12_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-link_update/dense_12/BiasAdd_3/ReadVariableOpÔ
link_update/dense_12/BiasAdd_3BiasAdd'link_update/dense_12/MatMul_3:product:05link_update/dense_12/BiasAdd_3/ReadVariableOp:value:0*
T0*
_output_shapes

:J2 
link_update/dense_12/BiasAdd_3
link_update/dense_12/Tanh_3Tanh'link_update/dense_12/BiasAdd_3:output:0*
T0*
_output_shapes

:J2
link_update/dense_12/Tanh_3	
GatherV2_8/indicesConst*
_output_shapes	
:*
dtype0*ą
value§B¤"                                                      
   
   
   
   
                                                                                                                                            !   !   !   !   !   !   "   "   "   "   "   "   "   "   #   #   #   %   %   %   %   %   %   &   &   &   '   '   '   '   '   '   '   '   (   (   (   (   )   )   )   )   )   )   )   )   *   *   +   +   +   +   +   +   ,   ,   ,   ,   -   -   -   -   -   -   -   -   .   .   .   .   /   /   /   0   0   0   0   0   1   1   1   1   1   1   1   1   2   2   2   2   3   3   3   3   4   4   4   4   4   5   5   5   6   6   6   6   7   7   7   8   8   8   8   8   8   9   9   9   9   9   9   9   9   :   :   :   :   ;   ;   ;   ;   <   <   =   =   =   =   >   >   >   >   ?   ?   ?   ?   ?   @   @   A   A   A   A   A   A   B   B   B   B   C   C   C   C   C   C   C   C   D   D   D   D   E   E   E   E   F   F   F   F   G   G   G   G   H   H   H   H   H   I   I   I   I   I   I   I   I   2
GatherV2_8/indicesd
GatherV2_8/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2_8/axisÉ

GatherV2_8GatherV2link_update/dense_12/Tanh_3:y:0GatherV2_8/indices:output:0GatherV2_8/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:	2

GatherV2_8	
GatherV2_9/indicesConst*
_output_shapes	
:*
dtype0*ą
value§B¤"      
                                  #   %   &   '   5   6   7   8   9          D   E   F   G   H   I                                                                         !   "                                 B   C   D   E   F   G   H   I   :   ;   <   =   >   ?   @   A         
   D   E   F   G   H   I   2   3   4   :   ;   <   =   >   ?   @   A   *   +   ,   -   :   ;   <   =   >   ?   @   A   (   )   D   E   F   G   H   I   .   /   0   1   :   ;   <   =   >   ?   @   A   *   +   ,   -   2   3   4   5   6   7   8   9   :   ;   <   =   >   ?   @   A   #   %   &   '   .   /   0   1   5   6   7   8   9         
   .   /   0   1   2   3   4   D   E   F   G   H   I   :   ;   <   =   >   ?   @   A          !   "   #   %   &   '   (   )   *   +   ,   -   .   /   0   1   5   6   7   8   9   B   C   D   E   F   G   H   I          !   "   :   ;   <   =   >   ?   @   A                      !   "   #   %   &   '   *   +   ,   -   5   6   7   8   9   :   ;   <   =   >   ?   @   A   2
GatherV2_9/indicesd
GatherV2_9/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2_9/axisÉ

GatherV2_9GatherV2link_update/dense_12/Tanh_3:y:0GatherV2_9/indices:output:0GatherV2_9/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:	2

GatherV2_9`
concat_8/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat_8/axis
concat_8ConcatV2GatherV2_8:output:0GatherV2_9:output:0concat_8/axis:output:0*
N*
T0*
_output_shapes
:	 2

concat_8Ö
.create_message/dense_8/MatMul_4/ReadVariableOpReadVariableOp5create_message_dense_8_matmul_readvariableop_resource*
_output_shapes

: @*
dtype020
.create_message/dense_8/MatMul_4/ReadVariableOpÁ
create_message/dense_8/MatMul_4MatMulconcat_8:output:06create_message/dense_8/MatMul_4/ReadVariableOp:value:0*
T0*
_output_shapes
:	@2!
create_message/dense_8/MatMul_4Ő
/create_message/dense_8/BiasAdd_4/ReadVariableOpReadVariableOp6create_message_dense_8_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype021
/create_message/dense_8/BiasAdd_4/ReadVariableOpÝ
 create_message/dense_8/BiasAdd_4BiasAdd)create_message/dense_8/MatMul_4:product:07create_message/dense_8/BiasAdd_4/ReadVariableOp:value:0*
T0*
_output_shapes
:	@2"
 create_message/dense_8/BiasAdd_4
create_message/dense_8/Tanh_4Tanh)create_message/dense_8/BiasAdd_4:output:0*
T0*
_output_shapes
:	@2
create_message/dense_8/Tanh_4Ö
.create_message/dense_9/MatMul_4/ReadVariableOpReadVariableOp5create_message_dense_9_matmul_readvariableop_resource*
_output_shapes

:@*
dtype020
.create_message/dense_9/MatMul_4/ReadVariableOpŃ
create_message/dense_9/MatMul_4MatMul!create_message/dense_8/Tanh_4:y:06create_message/dense_9/MatMul_4/ReadVariableOp:value:0*
T0*
_output_shapes
:	2!
create_message/dense_9/MatMul_4Ő
/create_message/dense_9/BiasAdd_4/ReadVariableOpReadVariableOp6create_message_dense_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype021
/create_message/dense_9/BiasAdd_4/ReadVariableOpÝ
 create_message/dense_9/BiasAdd_4BiasAdd)create_message/dense_9/MatMul_4:product:07create_message/dense_9/BiasAdd_4/ReadVariableOp:value:0*
T0*
_output_shapes
:	2"
 create_message/dense_9/BiasAdd_4
create_message/dense_9/Tanh_4Tanh)create_message/dense_9/BiasAdd_4:output:0*
T0*
_output_shapes
:	2
create_message/dense_9/Tanh_4ë
PartitionedCall_4PartitionedCall!create_message/dense_9/Tanh_4:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *Q
_output_shapes?
=:J : ::J:	: ::J: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *-
f(R&
$__inference_message_aggregation_12562
PartitionedCall_4`
concat_9/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat_9/axis§
concat_9ConcatV2link_update/dense_12/Tanh_3:y:0PartitionedCall_4:output:0concat_9/axis:output:0*
N*
T0*
_output_shapes

:J02

concat_9Ń
,link_update/dense_10/MatMul_4/ReadVariableOpReadVariableOp3link_update_dense_10_matmul_readvariableop_resource*
_output_shapes
:	0*
dtype02.
,link_update/dense_10/MatMul_4/ReadVariableOpť
link_update/dense_10/MatMul_4MatMulconcat_9:output:04link_update/dense_10/MatMul_4/ReadVariableOp:value:0*
T0*
_output_shapes
:	J2
link_update/dense_10/MatMul_4Đ
-link_update/dense_10/BiasAdd_4/ReadVariableOpReadVariableOp4link_update_dense_10_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02/
-link_update/dense_10/BiasAdd_4/ReadVariableOpŐ
link_update/dense_10/BiasAdd_4BiasAdd'link_update/dense_10/MatMul_4:product:05link_update/dense_10/BiasAdd_4/ReadVariableOp:value:0*
T0*
_output_shapes
:	J2 
link_update/dense_10/BiasAdd_4
link_update/dense_10/Tanh_4Tanh'link_update/dense_10/BiasAdd_4:output:0*
T0*
_output_shapes
:	J2
link_update/dense_10/Tanh_4Ń
,link_update/dense_11/MatMul_4/ReadVariableOpReadVariableOp3link_update_dense_11_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype02.
,link_update/dense_11/MatMul_4/ReadVariableOpČ
link_update/dense_11/MatMul_4MatMullink_update/dense_10/Tanh_4:y:04link_update/dense_11/MatMul_4/ReadVariableOp:value:0*
T0*
_output_shapes

:J@2
link_update/dense_11/MatMul_4Ď
-link_update/dense_11/BiasAdd_4/ReadVariableOpReadVariableOp4link_update_dense_11_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-link_update/dense_11/BiasAdd_4/ReadVariableOpÔ
link_update/dense_11/BiasAdd_4BiasAdd'link_update/dense_11/MatMul_4:product:05link_update/dense_11/BiasAdd_4/ReadVariableOp:value:0*
T0*
_output_shapes

:J@2 
link_update/dense_11/BiasAdd_4
link_update/dense_11/Tanh_4Tanh'link_update/dense_11/BiasAdd_4:output:0*
T0*
_output_shapes

:J@2
link_update/dense_11/Tanh_4Đ
,link_update/dense_12/MatMul_4/ReadVariableOpReadVariableOp3link_update_dense_12_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02.
,link_update/dense_12/MatMul_4/ReadVariableOpČ
link_update/dense_12/MatMul_4MatMullink_update/dense_11/Tanh_4:y:04link_update/dense_12/MatMul_4/ReadVariableOp:value:0*
T0*
_output_shapes

:J2
link_update/dense_12/MatMul_4Ď
-link_update/dense_12/BiasAdd_4/ReadVariableOpReadVariableOp4link_update_dense_12_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-link_update/dense_12/BiasAdd_4/ReadVariableOpÔ
link_update/dense_12/BiasAdd_4BiasAdd'link_update/dense_12/MatMul_4:product:05link_update/dense_12/BiasAdd_4/ReadVariableOp:value:0*
T0*
_output_shapes

:J2 
link_update/dense_12/BiasAdd_4
link_update/dense_12/Tanh_4Tanh'link_update/dense_12/BiasAdd_4:output:0*
T0*
_output_shapes

:J2
link_update/dense_12/Tanh_4	
GatherV2_10/indicesConst*
_output_shapes	
:*
dtype0*ą
value§B¤"                                                      
   
   
   
   
                                                                                                                                            !   !   !   !   !   !   "   "   "   "   "   "   "   "   #   #   #   %   %   %   %   %   %   &   &   &   '   '   '   '   '   '   '   '   (   (   (   (   )   )   )   )   )   )   )   )   *   *   +   +   +   +   +   +   ,   ,   ,   ,   -   -   -   -   -   -   -   -   .   .   .   .   /   /   /   0   0   0   0   0   1   1   1   1   1   1   1   1   2   2   2   2   3   3   3   3   4   4   4   4   4   5   5   5   6   6   6   6   7   7   7   8   8   8   8   8   8   9   9   9   9   9   9   9   9   :   :   :   :   ;   ;   ;   ;   <   <   =   =   =   =   >   >   >   >   ?   ?   ?   ?   ?   @   @   A   A   A   A   A   A   B   B   B   B   C   C   C   C   C   C   C   C   D   D   D   D   E   E   E   E   F   F   F   F   G   G   G   G   H   H   H   H   H   I   I   I   I   I   I   I   I   2
GatherV2_10/indicesf
GatherV2_10/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2_10/axisÍ
GatherV2_10GatherV2link_update/dense_12/Tanh_4:y:0GatherV2_10/indices:output:0GatherV2_10/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:	2
GatherV2_10	
GatherV2_11/indicesConst*
_output_shapes	
:*
dtype0*ą
value§B¤"      
                                  #   %   &   '   5   6   7   8   9          D   E   F   G   H   I                                                                         !   "                                 B   C   D   E   F   G   H   I   :   ;   <   =   >   ?   @   A         
   D   E   F   G   H   I   2   3   4   :   ;   <   =   >   ?   @   A   *   +   ,   -   :   ;   <   =   >   ?   @   A   (   )   D   E   F   G   H   I   .   /   0   1   :   ;   <   =   >   ?   @   A   *   +   ,   -   2   3   4   5   6   7   8   9   :   ;   <   =   >   ?   @   A   #   %   &   '   .   /   0   1   5   6   7   8   9         
   .   /   0   1   2   3   4   D   E   F   G   H   I   :   ;   <   =   >   ?   @   A          !   "   #   %   &   '   (   )   *   +   ,   -   .   /   0   1   5   6   7   8   9   B   C   D   E   F   G   H   I          !   "   :   ;   <   =   >   ?   @   A                      !   "   #   %   &   '   *   +   ,   -   5   6   7   8   9   :   ;   <   =   >   ?   @   A   2
GatherV2_11/indicesf
GatherV2_11/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2_11/axisÍ
GatherV2_11GatherV2link_update/dense_12/Tanh_4:y:0GatherV2_11/indices:output:0GatherV2_11/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:	2
GatherV2_11b
concat_10/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat_10/axis
	concat_10ConcatV2GatherV2_10:output:0GatherV2_11:output:0concat_10/axis:output:0*
N*
T0*
_output_shapes
:	 2
	concat_10Ö
.create_message/dense_8/MatMul_5/ReadVariableOpReadVariableOp5create_message_dense_8_matmul_readvariableop_resource*
_output_shapes

: @*
dtype020
.create_message/dense_8/MatMul_5/ReadVariableOpÂ
create_message/dense_8/MatMul_5MatMulconcat_10:output:06create_message/dense_8/MatMul_5/ReadVariableOp:value:0*
T0*
_output_shapes
:	@2!
create_message/dense_8/MatMul_5Ő
/create_message/dense_8/BiasAdd_5/ReadVariableOpReadVariableOp6create_message_dense_8_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype021
/create_message/dense_8/BiasAdd_5/ReadVariableOpÝ
 create_message/dense_8/BiasAdd_5BiasAdd)create_message/dense_8/MatMul_5:product:07create_message/dense_8/BiasAdd_5/ReadVariableOp:value:0*
T0*
_output_shapes
:	@2"
 create_message/dense_8/BiasAdd_5
create_message/dense_8/Tanh_5Tanh)create_message/dense_8/BiasAdd_5:output:0*
T0*
_output_shapes
:	@2
create_message/dense_8/Tanh_5Ö
.create_message/dense_9/MatMul_5/ReadVariableOpReadVariableOp5create_message_dense_9_matmul_readvariableop_resource*
_output_shapes

:@*
dtype020
.create_message/dense_9/MatMul_5/ReadVariableOpŃ
create_message/dense_9/MatMul_5MatMul!create_message/dense_8/Tanh_5:y:06create_message/dense_9/MatMul_5/ReadVariableOp:value:0*
T0*
_output_shapes
:	2!
create_message/dense_9/MatMul_5Ő
/create_message/dense_9/BiasAdd_5/ReadVariableOpReadVariableOp6create_message_dense_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype021
/create_message/dense_9/BiasAdd_5/ReadVariableOpÝ
 create_message/dense_9/BiasAdd_5BiasAdd)create_message/dense_9/MatMul_5:product:07create_message/dense_9/BiasAdd_5/ReadVariableOp:value:0*
T0*
_output_shapes
:	2"
 create_message/dense_9/BiasAdd_5
create_message/dense_9/Tanh_5Tanh)create_message/dense_9/BiasAdd_5:output:0*
T0*
_output_shapes
:	2
create_message/dense_9/Tanh_5ë
PartitionedCall_5PartitionedCall!create_message/dense_9/Tanh_5:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *Q
_output_shapes?
=:J : ::J:	: ::J: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *-
f(R&
$__inference_message_aggregation_12562
PartitionedCall_5b
concat_11/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat_11/axisŞ
	concat_11ConcatV2link_update/dense_12/Tanh_4:y:0PartitionedCall_5:output:0concat_11/axis:output:0*
N*
T0*
_output_shapes

:J02
	concat_11Ń
,link_update/dense_10/MatMul_5/ReadVariableOpReadVariableOp3link_update_dense_10_matmul_readvariableop_resource*
_output_shapes
:	0*
dtype02.
,link_update/dense_10/MatMul_5/ReadVariableOpź
link_update/dense_10/MatMul_5MatMulconcat_11:output:04link_update/dense_10/MatMul_5/ReadVariableOp:value:0*
T0*
_output_shapes
:	J2
link_update/dense_10/MatMul_5Đ
-link_update/dense_10/BiasAdd_5/ReadVariableOpReadVariableOp4link_update_dense_10_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02/
-link_update/dense_10/BiasAdd_5/ReadVariableOpŐ
link_update/dense_10/BiasAdd_5BiasAdd'link_update/dense_10/MatMul_5:product:05link_update/dense_10/BiasAdd_5/ReadVariableOp:value:0*
T0*
_output_shapes
:	J2 
link_update/dense_10/BiasAdd_5
link_update/dense_10/Tanh_5Tanh'link_update/dense_10/BiasAdd_5:output:0*
T0*
_output_shapes
:	J2
link_update/dense_10/Tanh_5Ń
,link_update/dense_11/MatMul_5/ReadVariableOpReadVariableOp3link_update_dense_11_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype02.
,link_update/dense_11/MatMul_5/ReadVariableOpČ
link_update/dense_11/MatMul_5MatMullink_update/dense_10/Tanh_5:y:04link_update/dense_11/MatMul_5/ReadVariableOp:value:0*
T0*
_output_shapes

:J@2
link_update/dense_11/MatMul_5Ď
-link_update/dense_11/BiasAdd_5/ReadVariableOpReadVariableOp4link_update_dense_11_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-link_update/dense_11/BiasAdd_5/ReadVariableOpÔ
link_update/dense_11/BiasAdd_5BiasAdd'link_update/dense_11/MatMul_5:product:05link_update/dense_11/BiasAdd_5/ReadVariableOp:value:0*
T0*
_output_shapes

:J@2 
link_update/dense_11/BiasAdd_5
link_update/dense_11/Tanh_5Tanh'link_update/dense_11/BiasAdd_5:output:0*
T0*
_output_shapes

:J@2
link_update/dense_11/Tanh_5Đ
,link_update/dense_12/MatMul_5/ReadVariableOpReadVariableOp3link_update_dense_12_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02.
,link_update/dense_12/MatMul_5/ReadVariableOpČ
link_update/dense_12/MatMul_5MatMullink_update/dense_11/Tanh_5:y:04link_update/dense_12/MatMul_5/ReadVariableOp:value:0*
T0*
_output_shapes

:J2
link_update/dense_12/MatMul_5Ď
-link_update/dense_12/BiasAdd_5/ReadVariableOpReadVariableOp4link_update_dense_12_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-link_update/dense_12/BiasAdd_5/ReadVariableOpÔ
link_update/dense_12/BiasAdd_5BiasAdd'link_update/dense_12/MatMul_5:product:05link_update/dense_12/BiasAdd_5/ReadVariableOp:value:0*
T0*
_output_shapes

:J2 
link_update/dense_12/BiasAdd_5
link_update/dense_12/Tanh_5Tanh'link_update/dense_12/BiasAdd_5:output:0*
T0*
_output_shapes

:J2
link_update/dense_12/Tanh_5	
GatherV2_12/indicesConst*
_output_shapes	
:*
dtype0*ą
value§B¤"                                                      
   
   
   
   
                                                                                                                                            !   !   !   !   !   !   "   "   "   "   "   "   "   "   #   #   #   %   %   %   %   %   %   &   &   &   '   '   '   '   '   '   '   '   (   (   (   (   )   )   )   )   )   )   )   )   *   *   +   +   +   +   +   +   ,   ,   ,   ,   -   -   -   -   -   -   -   -   .   .   .   .   /   /   /   0   0   0   0   0   1   1   1   1   1   1   1   1   2   2   2   2   3   3   3   3   4   4   4   4   4   5   5   5   6   6   6   6   7   7   7   8   8   8   8   8   8   9   9   9   9   9   9   9   9   :   :   :   :   ;   ;   ;   ;   <   <   =   =   =   =   >   >   >   >   ?   ?   ?   ?   ?   @   @   A   A   A   A   A   A   B   B   B   B   C   C   C   C   C   C   C   C   D   D   D   D   E   E   E   E   F   F   F   F   G   G   G   G   H   H   H   H   H   I   I   I   I   I   I   I   I   2
GatherV2_12/indicesf
GatherV2_12/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2_12/axisÍ
GatherV2_12GatherV2link_update/dense_12/Tanh_5:y:0GatherV2_12/indices:output:0GatherV2_12/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:	2
GatherV2_12	
GatherV2_13/indicesConst*
_output_shapes	
:*
dtype0*ą
value§B¤"      
                                  #   %   &   '   5   6   7   8   9          D   E   F   G   H   I                                                                         !   "                                 B   C   D   E   F   G   H   I   :   ;   <   =   >   ?   @   A         
   D   E   F   G   H   I   2   3   4   :   ;   <   =   >   ?   @   A   *   +   ,   -   :   ;   <   =   >   ?   @   A   (   )   D   E   F   G   H   I   .   /   0   1   :   ;   <   =   >   ?   @   A   *   +   ,   -   2   3   4   5   6   7   8   9   :   ;   <   =   >   ?   @   A   #   %   &   '   .   /   0   1   5   6   7   8   9         
   .   /   0   1   2   3   4   D   E   F   G   H   I   :   ;   <   =   >   ?   @   A          !   "   #   %   &   '   (   )   *   +   ,   -   .   /   0   1   5   6   7   8   9   B   C   D   E   F   G   H   I          !   "   :   ;   <   =   >   ?   @   A                      !   "   #   %   &   '   *   +   ,   -   5   6   7   8   9   :   ;   <   =   >   ?   @   A   2
GatherV2_13/indicesf
GatherV2_13/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2_13/axisÍ
GatherV2_13GatherV2link_update/dense_12/Tanh_5:y:0GatherV2_13/indices:output:0GatherV2_13/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:	2
GatherV2_13b
concat_12/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat_12/axis
	concat_12ConcatV2GatherV2_12:output:0GatherV2_13:output:0concat_12/axis:output:0*
N*
T0*
_output_shapes
:	 2
	concat_12Ö
.create_message/dense_8/MatMul_6/ReadVariableOpReadVariableOp5create_message_dense_8_matmul_readvariableop_resource*
_output_shapes

: @*
dtype020
.create_message/dense_8/MatMul_6/ReadVariableOpÂ
create_message/dense_8/MatMul_6MatMulconcat_12:output:06create_message/dense_8/MatMul_6/ReadVariableOp:value:0*
T0*
_output_shapes
:	@2!
create_message/dense_8/MatMul_6Ő
/create_message/dense_8/BiasAdd_6/ReadVariableOpReadVariableOp6create_message_dense_8_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype021
/create_message/dense_8/BiasAdd_6/ReadVariableOpÝ
 create_message/dense_8/BiasAdd_6BiasAdd)create_message/dense_8/MatMul_6:product:07create_message/dense_8/BiasAdd_6/ReadVariableOp:value:0*
T0*
_output_shapes
:	@2"
 create_message/dense_8/BiasAdd_6
create_message/dense_8/Tanh_6Tanh)create_message/dense_8/BiasAdd_6:output:0*
T0*
_output_shapes
:	@2
create_message/dense_8/Tanh_6Ö
.create_message/dense_9/MatMul_6/ReadVariableOpReadVariableOp5create_message_dense_9_matmul_readvariableop_resource*
_output_shapes

:@*
dtype020
.create_message/dense_9/MatMul_6/ReadVariableOpŃ
create_message/dense_9/MatMul_6MatMul!create_message/dense_8/Tanh_6:y:06create_message/dense_9/MatMul_6/ReadVariableOp:value:0*
T0*
_output_shapes
:	2!
create_message/dense_9/MatMul_6Ő
/create_message/dense_9/BiasAdd_6/ReadVariableOpReadVariableOp6create_message_dense_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype021
/create_message/dense_9/BiasAdd_6/ReadVariableOpÝ
 create_message/dense_9/BiasAdd_6BiasAdd)create_message/dense_9/MatMul_6:product:07create_message/dense_9/BiasAdd_6/ReadVariableOp:value:0*
T0*
_output_shapes
:	2"
 create_message/dense_9/BiasAdd_6
create_message/dense_9/Tanh_6Tanh)create_message/dense_9/BiasAdd_6:output:0*
T0*
_output_shapes
:	2
create_message/dense_9/Tanh_6ë
PartitionedCall_6PartitionedCall!create_message/dense_9/Tanh_6:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *Q
_output_shapes?
=:J : ::J:	: ::J: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *-
f(R&
$__inference_message_aggregation_12562
PartitionedCall_6b
concat_13/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat_13/axisŞ
	concat_13ConcatV2link_update/dense_12/Tanh_5:y:0PartitionedCall_6:output:0concat_13/axis:output:0*
N*
T0*
_output_shapes

:J02
	concat_13Ń
,link_update/dense_10/MatMul_6/ReadVariableOpReadVariableOp3link_update_dense_10_matmul_readvariableop_resource*
_output_shapes
:	0*
dtype02.
,link_update/dense_10/MatMul_6/ReadVariableOpź
link_update/dense_10/MatMul_6MatMulconcat_13:output:04link_update/dense_10/MatMul_6/ReadVariableOp:value:0*
T0*
_output_shapes
:	J2
link_update/dense_10/MatMul_6Đ
-link_update/dense_10/BiasAdd_6/ReadVariableOpReadVariableOp4link_update_dense_10_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02/
-link_update/dense_10/BiasAdd_6/ReadVariableOpŐ
link_update/dense_10/BiasAdd_6BiasAdd'link_update/dense_10/MatMul_6:product:05link_update/dense_10/BiasAdd_6/ReadVariableOp:value:0*
T0*
_output_shapes
:	J2 
link_update/dense_10/BiasAdd_6
link_update/dense_10/Tanh_6Tanh'link_update/dense_10/BiasAdd_6:output:0*
T0*
_output_shapes
:	J2
link_update/dense_10/Tanh_6Ń
,link_update/dense_11/MatMul_6/ReadVariableOpReadVariableOp3link_update_dense_11_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype02.
,link_update/dense_11/MatMul_6/ReadVariableOpČ
link_update/dense_11/MatMul_6MatMullink_update/dense_10/Tanh_6:y:04link_update/dense_11/MatMul_6/ReadVariableOp:value:0*
T0*
_output_shapes

:J@2
link_update/dense_11/MatMul_6Ď
-link_update/dense_11/BiasAdd_6/ReadVariableOpReadVariableOp4link_update_dense_11_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-link_update/dense_11/BiasAdd_6/ReadVariableOpÔ
link_update/dense_11/BiasAdd_6BiasAdd'link_update/dense_11/MatMul_6:product:05link_update/dense_11/BiasAdd_6/ReadVariableOp:value:0*
T0*
_output_shapes

:J@2 
link_update/dense_11/BiasAdd_6
link_update/dense_11/Tanh_6Tanh'link_update/dense_11/BiasAdd_6:output:0*
T0*
_output_shapes

:J@2
link_update/dense_11/Tanh_6Đ
,link_update/dense_12/MatMul_6/ReadVariableOpReadVariableOp3link_update_dense_12_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02.
,link_update/dense_12/MatMul_6/ReadVariableOpČ
link_update/dense_12/MatMul_6MatMullink_update/dense_11/Tanh_6:y:04link_update/dense_12/MatMul_6/ReadVariableOp:value:0*
T0*
_output_shapes

:J2
link_update/dense_12/MatMul_6Ď
-link_update/dense_12/BiasAdd_6/ReadVariableOpReadVariableOp4link_update_dense_12_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-link_update/dense_12/BiasAdd_6/ReadVariableOpÔ
link_update/dense_12/BiasAdd_6BiasAdd'link_update/dense_12/MatMul_6:product:05link_update/dense_12/BiasAdd_6/ReadVariableOp:value:0*
T0*
_output_shapes

:J2 
link_update/dense_12/BiasAdd_6
link_update/dense_12/Tanh_6Tanh'link_update/dense_12/BiasAdd_6:output:0*
T0*
_output_shapes

:J2
link_update/dense_12/Tanh_6	
GatherV2_14/indicesConst*
_output_shapes	
:*
dtype0*ą
value§B¤"                                                      
   
   
   
   
                                                                                                                                            !   !   !   !   !   !   "   "   "   "   "   "   "   "   #   #   #   %   %   %   %   %   %   &   &   &   '   '   '   '   '   '   '   '   (   (   (   (   )   )   )   )   )   )   )   )   *   *   +   +   +   +   +   +   ,   ,   ,   ,   -   -   -   -   -   -   -   -   .   .   .   .   /   /   /   0   0   0   0   0   1   1   1   1   1   1   1   1   2   2   2   2   3   3   3   3   4   4   4   4   4   5   5   5   6   6   6   6   7   7   7   8   8   8   8   8   8   9   9   9   9   9   9   9   9   :   :   :   :   ;   ;   ;   ;   <   <   =   =   =   =   >   >   >   >   ?   ?   ?   ?   ?   @   @   A   A   A   A   A   A   B   B   B   B   C   C   C   C   C   C   C   C   D   D   D   D   E   E   E   E   F   F   F   F   G   G   G   G   H   H   H   H   H   I   I   I   I   I   I   I   I   2
GatherV2_14/indicesf
GatherV2_14/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2_14/axisÍ
GatherV2_14GatherV2link_update/dense_12/Tanh_6:y:0GatherV2_14/indices:output:0GatherV2_14/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:	2
GatherV2_14	
GatherV2_15/indicesConst*
_output_shapes	
:*
dtype0*ą
value§B¤"      
                                  #   %   &   '   5   6   7   8   9          D   E   F   G   H   I                                                                         !   "                                 B   C   D   E   F   G   H   I   :   ;   <   =   >   ?   @   A         
   D   E   F   G   H   I   2   3   4   :   ;   <   =   >   ?   @   A   *   +   ,   -   :   ;   <   =   >   ?   @   A   (   )   D   E   F   G   H   I   .   /   0   1   :   ;   <   =   >   ?   @   A   *   +   ,   -   2   3   4   5   6   7   8   9   :   ;   <   =   >   ?   @   A   #   %   &   '   .   /   0   1   5   6   7   8   9         
   .   /   0   1   2   3   4   D   E   F   G   H   I   :   ;   <   =   >   ?   @   A          !   "   #   %   &   '   (   )   *   +   ,   -   .   /   0   1   5   6   7   8   9   B   C   D   E   F   G   H   I          !   "   :   ;   <   =   >   ?   @   A                      !   "   #   %   &   '   *   +   ,   -   5   6   7   8   9   :   ;   <   =   >   ?   @   A   2
GatherV2_15/indicesf
GatherV2_15/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2_15/axisÍ
GatherV2_15GatherV2link_update/dense_12/Tanh_6:y:0GatherV2_15/indices:output:0GatherV2_15/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:	2
GatherV2_15b
concat_14/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat_14/axis
	concat_14ConcatV2GatherV2_14:output:0GatherV2_15:output:0concat_14/axis:output:0*
N*
T0*
_output_shapes
:	 2
	concat_14Ö
.create_message/dense_8/MatMul_7/ReadVariableOpReadVariableOp5create_message_dense_8_matmul_readvariableop_resource*
_output_shapes

: @*
dtype020
.create_message/dense_8/MatMul_7/ReadVariableOpÂ
create_message/dense_8/MatMul_7MatMulconcat_14:output:06create_message/dense_8/MatMul_7/ReadVariableOp:value:0*
T0*
_output_shapes
:	@2!
create_message/dense_8/MatMul_7Ő
/create_message/dense_8/BiasAdd_7/ReadVariableOpReadVariableOp6create_message_dense_8_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype021
/create_message/dense_8/BiasAdd_7/ReadVariableOpÝ
 create_message/dense_8/BiasAdd_7BiasAdd)create_message/dense_8/MatMul_7:product:07create_message/dense_8/BiasAdd_7/ReadVariableOp:value:0*
T0*
_output_shapes
:	@2"
 create_message/dense_8/BiasAdd_7
create_message/dense_8/Tanh_7Tanh)create_message/dense_8/BiasAdd_7:output:0*
T0*
_output_shapes
:	@2
create_message/dense_8/Tanh_7Ö
.create_message/dense_9/MatMul_7/ReadVariableOpReadVariableOp5create_message_dense_9_matmul_readvariableop_resource*
_output_shapes

:@*
dtype020
.create_message/dense_9/MatMul_7/ReadVariableOpŃ
create_message/dense_9/MatMul_7MatMul!create_message/dense_8/Tanh_7:y:06create_message/dense_9/MatMul_7/ReadVariableOp:value:0*
T0*
_output_shapes
:	2!
create_message/dense_9/MatMul_7Ő
/create_message/dense_9/BiasAdd_7/ReadVariableOpReadVariableOp6create_message_dense_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype021
/create_message/dense_9/BiasAdd_7/ReadVariableOpÝ
 create_message/dense_9/BiasAdd_7BiasAdd)create_message/dense_9/MatMul_7:product:07create_message/dense_9/BiasAdd_7/ReadVariableOp:value:0*
T0*
_output_shapes
:	2"
 create_message/dense_9/BiasAdd_7
create_message/dense_9/Tanh_7Tanh)create_message/dense_9/BiasAdd_7:output:0*
T0*
_output_shapes
:	2
create_message/dense_9/Tanh_7ë
PartitionedCall_7PartitionedCall!create_message/dense_9/Tanh_7:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *Q
_output_shapes?
=:J : ::J:	: ::J: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *-
f(R&
$__inference_message_aggregation_12562
PartitionedCall_7b
concat_15/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat_15/axisŞ
	concat_15ConcatV2link_update/dense_12/Tanh_6:y:0PartitionedCall_7:output:0concat_15/axis:output:0*
N*
T0*
_output_shapes

:J02
	concat_15Ń
,link_update/dense_10/MatMul_7/ReadVariableOpReadVariableOp3link_update_dense_10_matmul_readvariableop_resource*
_output_shapes
:	0*
dtype02.
,link_update/dense_10/MatMul_7/ReadVariableOpź
link_update/dense_10/MatMul_7MatMulconcat_15:output:04link_update/dense_10/MatMul_7/ReadVariableOp:value:0*
T0*
_output_shapes
:	J2
link_update/dense_10/MatMul_7Đ
-link_update/dense_10/BiasAdd_7/ReadVariableOpReadVariableOp4link_update_dense_10_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02/
-link_update/dense_10/BiasAdd_7/ReadVariableOpŐ
link_update/dense_10/BiasAdd_7BiasAdd'link_update/dense_10/MatMul_7:product:05link_update/dense_10/BiasAdd_7/ReadVariableOp:value:0*
T0*
_output_shapes
:	J2 
link_update/dense_10/BiasAdd_7
link_update/dense_10/Tanh_7Tanh'link_update/dense_10/BiasAdd_7:output:0*
T0*
_output_shapes
:	J2
link_update/dense_10/Tanh_7Ń
,link_update/dense_11/MatMul_7/ReadVariableOpReadVariableOp3link_update_dense_11_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype02.
,link_update/dense_11/MatMul_7/ReadVariableOpČ
link_update/dense_11/MatMul_7MatMullink_update/dense_10/Tanh_7:y:04link_update/dense_11/MatMul_7/ReadVariableOp:value:0*
T0*
_output_shapes

:J@2
link_update/dense_11/MatMul_7Ď
-link_update/dense_11/BiasAdd_7/ReadVariableOpReadVariableOp4link_update_dense_11_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-link_update/dense_11/BiasAdd_7/ReadVariableOpÔ
link_update/dense_11/BiasAdd_7BiasAdd'link_update/dense_11/MatMul_7:product:05link_update/dense_11/BiasAdd_7/ReadVariableOp:value:0*
T0*
_output_shapes

:J@2 
link_update/dense_11/BiasAdd_7
link_update/dense_11/Tanh_7Tanh'link_update/dense_11/BiasAdd_7:output:0*
T0*
_output_shapes

:J@2
link_update/dense_11/Tanh_7Đ
,link_update/dense_12/MatMul_7/ReadVariableOpReadVariableOp3link_update_dense_12_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02.
,link_update/dense_12/MatMul_7/ReadVariableOpČ
link_update/dense_12/MatMul_7MatMullink_update/dense_11/Tanh_7:y:04link_update/dense_12/MatMul_7/ReadVariableOp:value:0*
T0*
_output_shapes

:J2
link_update/dense_12/MatMul_7Ď
-link_update/dense_12/BiasAdd_7/ReadVariableOpReadVariableOp4link_update_dense_12_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-link_update/dense_12/BiasAdd_7/ReadVariableOpÔ
link_update/dense_12/BiasAdd_7BiasAdd'link_update/dense_12/MatMul_7:product:05link_update/dense_12/BiasAdd_7/ReadVariableOp:value:0*
T0*
_output_shapes

:J2 
link_update/dense_12/BiasAdd_7
link_update/dense_12/Tanh_7Tanh'link_update/dense_12/BiasAdd_7:output:0*
T0*
_output_shapes

:J2
link_update/dense_12/Tanh_7j
IdentityIdentitylink_update/dense_12/Tanh_7:y:0*
T0*
_output_shapes

:J2

Identity"
identityIdentity:output:0*B
_input_shapes1
/::::::::::::B >

_output_shapes	
:

_user_specified_nameinput
ń
Ŕ
'__inference_readout_layer_call_fn_54035
dense_13_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity˘StatefulPartitionedCallŽ
StatefulPartitionedCallStatefulPartitionedCalldense_13_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_readout_layer_call_and_return_conditional_losses_540202
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:˙˙˙˙˙˙˙˙˙@::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
'
_output_shapes
:˙˙˙˙˙˙˙˙˙@
(
_user_specified_namedense_13_input
Ç"
¤
#__forward_message_aggregation_20599

messages_0
identity
concat_axis"
unsortedsegmentmax_segment_ids
unsortedsegmentmax
messages#
unsortedsegmentmax_num_segments"
unsortedsegmentmin_segment_ids
unsortedsegmentmin#
unsortedsegmentmin_num_segments§	
UnsortedSegmentMax/segment_idsConst*
_output_shapes	
:*
dtype0*ą
value§B¤"      
                                  #   %   &   '   5   6   7   8   9          D   E   F   G   H   I                                                                         !   "                                 B   C   D   E   F   G   H   I   :   ;   <   =   >   ?   @   A         
   D   E   F   G   H   I   2   3   4   :   ;   <   =   >   ?   @   A   *   +   ,   -   :   ;   <   =   >   ?   @   A   (   )   D   E   F   G   H   I   .   /   0   1   :   ;   <   =   >   ?   @   A   *   +   ,   -   2   3   4   5   6   7   8   9   :   ;   <   =   >   ?   @   A   #   %   &   '   .   /   0   1   5   6   7   8   9         
   .   /   0   1   2   3   4   D   E   F   G   H   I   :   ;   <   =   >   ?   @   A          !   "   #   %   &   '   (   )   *   +   ,   -   .   /   0   1   5   6   7   8   9   B   C   D   E   F   G   H   I          !   "   :   ;   <   =   >   ?   @   A                      !   "   #   %   &   '   *   +   ,   -   5   6   7   8   9   :   ;   <   =   >   ?   @   A   2 
UnsortedSegmentMax/segment_ids
UnsortedSegmentMax/num_segmentsConst*
_output_shapes
: *
dtype0*
value	B :J2!
UnsortedSegmentMax/num_segmentsÖ
UnsortedSegmentMaxUnsortedSegmentMax
messages_0'UnsortedSegmentMax/segment_ids:output:0(UnsortedSegmentMax/num_segments:output:0*
T0*
Tindices0*
_output_shapes

:J2
UnsortedSegmentMax§	
UnsortedSegmentMin/segment_idsConst*
_output_shapes	
:*
dtype0*ą
value§B¤"      
                                  #   %   &   '   5   6   7   8   9          D   E   F   G   H   I                                                                         !   "                                 B   C   D   E   F   G   H   I   :   ;   <   =   >   ?   @   A         
   D   E   F   G   H   I   2   3   4   :   ;   <   =   >   ?   @   A   *   +   ,   -   :   ;   <   =   >   ?   @   A   (   )   D   E   F   G   H   I   .   /   0   1   :   ;   <   =   >   ?   @   A   *   +   ,   -   2   3   4   5   6   7   8   9   :   ;   <   =   >   ?   @   A   #   %   &   '   .   /   0   1   5   6   7   8   9         
   .   /   0   1   2   3   4   D   E   F   G   H   I   :   ;   <   =   >   ?   @   A          !   "   #   %   &   '   (   )   *   +   ,   -   .   /   0   1   5   6   7   8   9   B   C   D   E   F   G   H   I          !   "   :   ;   <   =   >   ?   @   A                      !   "   #   %   &   '   *   +   ,   -   5   6   7   8   9   :   ;   <   =   >   ?   @   A   2 
UnsortedSegmentMin/segment_ids
UnsortedSegmentMin/num_segmentsConst*
_output_shapes
: *
dtype0*
value	B :J2!
UnsortedSegmentMin/num_segmentsÖ
UnsortedSegmentMinUnsortedSegmentMin
messages_0'UnsortedSegmentMin/segment_ids:output:0(UnsortedSegmentMin/num_segments:output:0*
T0*
Tindices0*
_output_shapes

:J2
UnsortedSegmentMin\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis
concatConcatV2UnsortedSegmentMax:output:0UnsortedSegmentMin:output:0concat/axis:output:0*
N*
T0*
_output_shapes

:J 2
concatZ
IdentityIdentityconcat:output:0*
T0*
_output_shapes

:J 2

Identity"#
concat_axisconcat/axis:output:0"
identityIdentity:output:0"
messages
messages_0"1
unsortedsegmentmaxUnsortedSegmentMax:output:0"K
unsortedsegmentmax_num_segments(UnsortedSegmentMax/num_segments:output:0"I
unsortedsegmentmax_segment_ids'UnsortedSegmentMax/segment_ids:output:0"1
unsortedsegmentminUnsortedSegmentMin:output:0"K
unsortedsegmentmin_num_segments(UnsortedSegmentMin/num_segments:output:0"I
unsortedsegmentmin_segment_ids'UnsortedSegmentMin/segment_ids:output:0*
_input_shapes
:	*R
backward_function_name86__inference___backward_message_aggregation_20495_20600:I E

_output_shapes
:	
"
_user_specified_name
messages
ˇ*
×
__inference__traced_save_55417
file_prefix-
)savev2_dense_8_kernel_read_readvariableop+
'savev2_dense_8_bias_read_readvariableop-
)savev2_dense_9_kernel_read_readvariableop+
'savev2_dense_9_bias_read_readvariableop.
*savev2_dense_10_kernel_read_readvariableop,
(savev2_dense_10_bias_read_readvariableop.
*savev2_dense_11_kernel_read_readvariableop,
(savev2_dense_11_bias_read_readvariableop.
*savev2_dense_12_kernel_read_readvariableop,
(savev2_dense_12_bias_read_readvariableop.
*savev2_dense_13_kernel_read_readvariableop,
(savev2_dense_13_bias_read_readvariableop.
*savev2_dense_14_kernel_read_readvariableop,
(savev2_dense_14_bias_read_readvariableop.
*savev2_dense_15_kernel_read_readvariableop,
(savev2_dense_15_bias_read_readvariableop
savev2_const

identity_1˘MergeV2Checkpoints
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Const
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_b45a493fc1874f9f8fd1c9f3e7ef553f/part2	
Const_1
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shardŚ
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameÇ
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Ů
valueĎBĚB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/13/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/14/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/15/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesŞ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*5
value,B*B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesö
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0)savev2_dense_8_kernel_read_readvariableop'savev2_dense_8_bias_read_readvariableop)savev2_dense_9_kernel_read_readvariableop'savev2_dense_9_bias_read_readvariableop*savev2_dense_10_kernel_read_readvariableop(savev2_dense_10_bias_read_readvariableop*savev2_dense_11_kernel_read_readvariableop(savev2_dense_11_bias_read_readvariableop*savev2_dense_12_kernel_read_readvariableop(savev2_dense_12_bias_read_readvariableop*savev2_dense_13_kernel_read_readvariableop(savev2_dense_13_bias_read_readvariableop*savev2_dense_14_kernel_read_readvariableop(savev2_dense_14_bias_read_readvariableop*savev2_dense_15_kernel_read_readvariableop(savev2_dense_15_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
22
SaveV2ş
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixesĄ
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*
_input_shapes
: : @:@:@::	0::	@:@:@::	@::	@:@:@:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

: @: 

_output_shapes
:@:$ 

_output_shapes

:@: 

_output_shapes
::%!

_output_shapes
:	0:!

_output_shapes	
::%!

_output_shapes
:	@: 

_output_shapes
:@:$	 

_output_shapes

:@: 


_output_shapes
::%!

_output_shapes
:	@:!

_output_shapes	
::%!

_output_shapes
:	@: 

_output_shapes
:@:$ 

_output_shapes

:@: 

_output_shapes
::

_output_shapes
: 
ě
H
'__inference_generate_readout_input_1577
link_states
identityr
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 2
Mean/reduction_indicesg
MeanMeanlink_statesMean/reduction_indices:output:0*
T0*
_output_shapes
:2
Meanp
Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 2
Max/reduction_indicesc
MaxMaxlink_statesMax/reduction_indices:output:0*
T0*
_output_shapes
:2
Maxp
Min/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 2
Min/reduction_indicesc
MinMinlink_statesMin/reduction_indices:output:0*
T0*
_output_shapes
:2
Min¨
1reduce_std/reduce_variance/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 23
1reduce_std/reduce_variance/Mean/reduction_indicesÍ
reduce_std/reduce_variance/MeanMeanlink_states:reduce_std/reduce_variance/Mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2!
reduce_std/reduce_variance/Mean§
reduce_std/reduce_variance/subSublink_states(reduce_std/reduce_variance/Mean:output:0*
T0*
_output_shapes

:J2 
reduce_std/reduce_variance/sub
!reduce_std/reduce_variance/SquareSquare"reduce_std/reduce_variance/sub:z:0*
T0*
_output_shapes

:J2#
!reduce_std/reduce_variance/SquareŹ
3reduce_std/reduce_variance/Mean_1/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 25
3reduce_std/reduce_variance/Mean_1/reduction_indicesŘ
!reduce_std/reduce_variance/Mean_1Mean%reduce_std/reduce_variance/Square:y:0<reduce_std/reduce_variance/Mean_1/reduction_indices:output:0*
T0*
_output_shapes
:2#
!reduce_std/reduce_variance/Mean_1{
reduce_std/SqrtSqrt*reduce_std/reduce_variance/Mean_1:output:0*
T0*
_output_shapes
:2
reduce_std/Sqrt\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat/axis 
concatConcatV2Mean:output:0Max:output:0Min:output:0reduce_std/Sqrt:y:0concat/axis:output:0*
N*
T0*
_output_shapes
:@2
concatb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims/dimy

ExpandDims
ExpandDimsconcat:output:0ExpandDims/dim:output:0*
T0*
_output_shapes

:@2

ExpandDims^
IdentityIdentityExpandDims:output:0*
T0*
_output_shapes

:@2

Identity"
identityIdentity:output:0*
_input_shapes

:J:K G

_output_shapes

:J
%
_user_specified_namelink_states

E
)__inference_dropout_3_layer_call_fn_55327

inputs
identityÂ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_3_layer_call_and_return_conditional_losses_539352
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@2

Identity"
identityIdentity:output:0*&
_input_shapes
:˙˙˙˙˙˙˙˙˙@:O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙@
 
_user_specified_nameinputs
š:
Đ
A__inference_critic_layer_call_and_return_conditional_losses_54475
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_83
/readout_dense_13_matmul_readvariableop_resource4
0readout_dense_13_biasadd_readvariableop_resource3
/readout_dense_14_matmul_readvariableop_resource4
0readout_dense_14_biasadd_readvariableop_resource3
/readout_dense_15_matmul_readvariableop_resource4
0readout_dense_15_biasadd_readvariableop_resource
identity˘StatefulPartitionedCallą
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:J*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8 **
f%R#
!__inference_message_passing_424792
StatefulPartitionedCallś
PartitionedCallPartitionedCall StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *0
f+R)
'__inference_generate_readout_input_15772
PartitionedCallÁ
&readout/dense_13/MatMul/ReadVariableOpReadVariableOp/readout_dense_13_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype02(
&readout/dense_13/MatMul/ReadVariableOp°
readout/dense_13/MatMulMatMulPartitionedCall:output:0.readout/dense_13/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
readout/dense_13/MatMulŔ
'readout/dense_13/BiasAdd/ReadVariableOpReadVariableOp0readout_dense_13_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02)
'readout/dense_13/BiasAdd/ReadVariableOp˝
readout/dense_13/BiasAddBiasAdd!readout/dense_13/MatMul:product:0/readout/dense_13/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
readout/dense_13/BiasAdd
readout/dense_13/TanhTanh!readout/dense_13/BiasAdd:output:0*
T0*
_output_shapes
:	2
readout/dense_13/Tanh
readout/dropout_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?2!
readout/dropout_2/dropout/Const´
readout/dropout_2/dropout/MulMulreadout/dense_13/Tanh:y:0(readout/dropout_2/dropout/Const:output:0*
T0*
_output_shapes
:	2
readout/dropout_2/dropout/Mul
readout/dropout_2/dropout/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2!
readout/dropout_2/dropout/Shapeâ
6readout/dropout_2/dropout/random_uniform/RandomUniformRandomUniform(readout/dropout_2/dropout/Shape:output:0*
T0*
_output_shapes
:	*
dtype028
6readout/dropout_2/dropout/random_uniform/RandomUniform
(readout/dropout_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2*
(readout/dropout_2/dropout/GreaterEqual/yţ
&readout/dropout_2/dropout/GreaterEqualGreaterEqual?readout/dropout_2/dropout/random_uniform/RandomUniform:output:01readout/dropout_2/dropout/GreaterEqual/y:output:0*
T0*
_output_shapes
:	2(
&readout/dropout_2/dropout/GreaterEqual­
readout/dropout_2/dropout/CastCast*readout/dropout_2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*
_output_shapes
:	2 
readout/dropout_2/dropout/Castş
readout/dropout_2/dropout/Mul_1Mul!readout/dropout_2/dropout/Mul:z:0"readout/dropout_2/dropout/Cast:y:0*
T0*
_output_shapes
:	2!
readout/dropout_2/dropout/Mul_1Á
&readout/dense_14/MatMul/ReadVariableOpReadVariableOp/readout_dense_14_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype02(
&readout/dense_14/MatMul/ReadVariableOpş
readout/dense_14/MatMulMatMul#readout/dropout_2/dropout/Mul_1:z:0.readout/dense_14/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:@2
readout/dense_14/MatMulż
'readout/dense_14/BiasAdd/ReadVariableOpReadVariableOp0readout_dense_14_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02)
'readout/dense_14/BiasAdd/ReadVariableOpź
readout/dense_14/BiasAddBiasAdd!readout/dense_14/MatMul:product:0/readout/dense_14/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:@2
readout/dense_14/BiasAdd
readout/dense_14/TanhTanh!readout/dense_14/BiasAdd:output:0*
T0*
_output_shapes

:@2
readout/dense_14/Tanh
readout/dropout_3/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?2!
readout/dropout_3/dropout/Constł
readout/dropout_3/dropout/MulMulreadout/dense_14/Tanh:y:0(readout/dropout_3/dropout/Const:output:0*
T0*
_output_shapes

:@2
readout/dropout_3/dropout/Mul
readout/dropout_3/dropout/ShapeConst*
_output_shapes
:*
dtype0*
valueB"   @   2!
readout/dropout_3/dropout/Shapeá
6readout/dropout_3/dropout/random_uniform/RandomUniformRandomUniform(readout/dropout_3/dropout/Shape:output:0*
T0*
_output_shapes

:@*
dtype028
6readout/dropout_3/dropout/random_uniform/RandomUniform
(readout/dropout_3/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2*
(readout/dropout_3/dropout/GreaterEqual/yý
&readout/dropout_3/dropout/GreaterEqualGreaterEqual?readout/dropout_3/dropout/random_uniform/RandomUniform:output:01readout/dropout_3/dropout/GreaterEqual/y:output:0*
T0*
_output_shapes

:@2(
&readout/dropout_3/dropout/GreaterEqualŹ
readout/dropout_3/dropout/CastCast*readout/dropout_3/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*
_output_shapes

:@2 
readout/dropout_3/dropout/Castš
readout/dropout_3/dropout/Mul_1Mul!readout/dropout_3/dropout/Mul:z:0"readout/dropout_3/dropout/Cast:y:0*
T0*
_output_shapes

:@2!
readout/dropout_3/dropout/Mul_1Ŕ
&readout/dense_15/MatMul/ReadVariableOpReadVariableOp/readout_dense_15_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02(
&readout/dense_15/MatMul/ReadVariableOpş
readout/dense_15/MatMulMatMul#readout/dropout_3/dropout/Mul_1:z:0.readout/dense_15/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
readout/dense_15/MatMulż
'readout/dense_15/BiasAdd/ReadVariableOpReadVariableOp0readout_dense_15_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'readout/dense_15/BiasAdd/ReadVariableOpź
readout/dense_15/BiasAddBiasAdd!readout/dense_15/MatMul:product:0/readout/dense_15/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:2
readout/dense_15/BiasAddq
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙2
Reshape/shape}
ReshapeReshape!readout/dense_15/BiasAdd:output:0Reshape/shape:output:0*
T0*
_output_shapes
:2	
Reshapeq
IdentityIdentityReshape:output:0^StatefulPartitionedCall*
T0*
_output_shapes
:2

Identity"
identityIdentity:output:0*b
_input_shapesQ
O:˙˙˙˙˙˙˙˙˙::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:L H
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
!
_user_specified_name	input_1
Ç"
¤
#__forward_message_aggregation_22691

messages_0
identity
concat_axis"
unsortedsegmentmax_segment_ids
unsortedsegmentmax
messages#
unsortedsegmentmax_num_segments"
unsortedsegmentmin_segment_ids
unsortedsegmentmin#
unsortedsegmentmin_num_segments§	
UnsortedSegmentMax/segment_idsConst*
_output_shapes	
:*
dtype0*ą
value§B¤"      
                                  #   %   &   '   5   6   7   8   9          D   E   F   G   H   I                                                                         !   "                                 B   C   D   E   F   G   H   I   :   ;   <   =   >   ?   @   A         
   D   E   F   G   H   I   2   3   4   :   ;   <   =   >   ?   @   A   *   +   ,   -   :   ;   <   =   >   ?   @   A   (   )   D   E   F   G   H   I   .   /   0   1   :   ;   <   =   >   ?   @   A   *   +   ,   -   2   3   4   5   6   7   8   9   :   ;   <   =   >   ?   @   A   #   %   &   '   .   /   0   1   5   6   7   8   9         
   .   /   0   1   2   3   4   D   E   F   G   H   I   :   ;   <   =   >   ?   @   A          !   "   #   %   &   '   (   )   *   +   ,   -   .   /   0   1   5   6   7   8   9   B   C   D   E   F   G   H   I          !   "   :   ;   <   =   >   ?   @   A                      !   "   #   %   &   '   *   +   ,   -   5   6   7   8   9   :   ;   <   =   >   ?   @   A   2 
UnsortedSegmentMax/segment_ids
UnsortedSegmentMax/num_segmentsConst*
_output_shapes
: *
dtype0*
value	B :J2!
UnsortedSegmentMax/num_segmentsÖ
UnsortedSegmentMaxUnsortedSegmentMax
messages_0'UnsortedSegmentMax/segment_ids:output:0(UnsortedSegmentMax/num_segments:output:0*
T0*
Tindices0*
_output_shapes

:J2
UnsortedSegmentMax§	
UnsortedSegmentMin/segment_idsConst*
_output_shapes	
:*
dtype0*ą
value§B¤"      
                                  #   %   &   '   5   6   7   8   9          D   E   F   G   H   I                                                                         !   "                                 B   C   D   E   F   G   H   I   :   ;   <   =   >   ?   @   A         
   D   E   F   G   H   I   2   3   4   :   ;   <   =   >   ?   @   A   *   +   ,   -   :   ;   <   =   >   ?   @   A   (   )   D   E   F   G   H   I   .   /   0   1   :   ;   <   =   >   ?   @   A   *   +   ,   -   2   3   4   5   6   7   8   9   :   ;   <   =   >   ?   @   A   #   %   &   '   .   /   0   1   5   6   7   8   9         
   .   /   0   1   2   3   4   D   E   F   G   H   I   :   ;   <   =   >   ?   @   A          !   "   #   %   &   '   (   )   *   +   ,   -   .   /   0   1   5   6   7   8   9   B   C   D   E   F   G   H   I          !   "   :   ;   <   =   >   ?   @   A                      !   "   #   %   &   '   *   +   ,   -   5   6   7   8   9   :   ;   <   =   >   ?   @   A   2 
UnsortedSegmentMin/segment_ids
UnsortedSegmentMin/num_segmentsConst*
_output_shapes
: *
dtype0*
value	B :J2!
UnsortedSegmentMin/num_segmentsÖ
UnsortedSegmentMinUnsortedSegmentMin
messages_0'UnsortedSegmentMin/segment_ids:output:0(UnsortedSegmentMin/num_segments:output:0*
T0*
Tindices0*
_output_shapes

:J2
UnsortedSegmentMin\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis
concatConcatV2UnsortedSegmentMax:output:0UnsortedSegmentMin:output:0concat/axis:output:0*
N*
T0*
_output_shapes

:J 2
concatZ
IdentityIdentityconcat:output:0*
T0*
_output_shapes

:J 2

Identity"#
concat_axisconcat/axis:output:0"
identityIdentity:output:0"
messages
messages_0"1
unsortedsegmentmaxUnsortedSegmentMax:output:0"K
unsortedsegmentmax_num_segments(UnsortedSegmentMax/num_segments:output:0"I
unsortedsegmentmax_segment_ids'UnsortedSegmentMax/segment_ids:output:0"1
unsortedsegmentminUnsortedSegmentMin:output:0"K
unsortedsegmentmin_num_segments(UnsortedSegmentMin/num_segments:output:0"I
unsortedsegmentmin_segment_ids'UnsortedSegmentMin/segment_ids:output:0*
_input_shapes
:	*R
backward_function_name86__inference___backward_message_aggregation_22579_22692:I E

_output_shapes
:	
"
_user_specified_name
messages
Ď
Î
F__inference_link_update_layer_call_and_return_conditional_losses_53757
dense_10_input
dense_10_53741
dense_10_53743
dense_11_53746
dense_11_53748
dense_12_53751
dense_12_53753
identity˘ dense_10/StatefulPartitionedCall˘ dense_11/StatefulPartitionedCall˘ dense_12/StatefulPartitionedCall
 dense_10/StatefulPartitionedCallStatefulPartitionedCalldense_10_inputdense_10_53741dense_10_53743*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_10_layer_call_and_return_conditional_losses_536672"
 dense_10/StatefulPartitionedCall´
 dense_11/StatefulPartitionedCallStatefulPartitionedCall)dense_10/StatefulPartitionedCall:output:0dense_11_53746dense_11_53748*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_11_layer_call_and_return_conditional_losses_536942"
 dense_11/StatefulPartitionedCall´
 dense_12/StatefulPartitionedCallStatefulPartitionedCall)dense_11/StatefulPartitionedCall:output:0dense_12_53751dense_12_53753*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_12_layer_call_and_return_conditional_losses_537212"
 dense_12/StatefulPartitionedCallć
IdentityIdentity)dense_12/StatefulPartitionedCall:output:0!^dense_10/StatefulPartitionedCall!^dense_11/StatefulPartitionedCall!^dense_12/StatefulPartitionedCall*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:˙˙˙˙˙˙˙˙˙0::::::2D
 dense_10/StatefulPartitionedCall dense_10/StatefulPartitionedCall2D
 dense_11/StatefulPartitionedCall dense_11/StatefulPartitionedCall2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall:W S
'
_output_shapes
:˙˙˙˙˙˙˙˙˙0
(
_user_specified_namedense_10_input
ł:
Î
A__inference_critic_layer_call_and_return_conditional_losses_54663	
input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_83
/readout_dense_13_matmul_readvariableop_resource4
0readout_dense_13_biasadd_readvariableop_resource3
/readout_dense_14_matmul_readvariableop_resource4
0readout_dense_14_biasadd_readvariableop_resource3
/readout_dense_15_matmul_readvariableop_resource4
0readout_dense_15_biasadd_readvariableop_resource
identity˘StatefulPartitionedCallŻ
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:J*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8 **
f%R#
!__inference_message_passing_424792
StatefulPartitionedCallś
PartitionedCallPartitionedCall StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *0
f+R)
'__inference_generate_readout_input_15772
PartitionedCallÁ
&readout/dense_13/MatMul/ReadVariableOpReadVariableOp/readout_dense_13_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype02(
&readout/dense_13/MatMul/ReadVariableOp°
readout/dense_13/MatMulMatMulPartitionedCall:output:0.readout/dense_13/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
readout/dense_13/MatMulŔ
'readout/dense_13/BiasAdd/ReadVariableOpReadVariableOp0readout_dense_13_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02)
'readout/dense_13/BiasAdd/ReadVariableOp˝
readout/dense_13/BiasAddBiasAdd!readout/dense_13/MatMul:product:0/readout/dense_13/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
readout/dense_13/BiasAdd
readout/dense_13/TanhTanh!readout/dense_13/BiasAdd:output:0*
T0*
_output_shapes
:	2
readout/dense_13/Tanh
readout/dropout_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?2!
readout/dropout_2/dropout/Const´
readout/dropout_2/dropout/MulMulreadout/dense_13/Tanh:y:0(readout/dropout_2/dropout/Const:output:0*
T0*
_output_shapes
:	2
readout/dropout_2/dropout/Mul
readout/dropout_2/dropout/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2!
readout/dropout_2/dropout/Shapeâ
6readout/dropout_2/dropout/random_uniform/RandomUniformRandomUniform(readout/dropout_2/dropout/Shape:output:0*
T0*
_output_shapes
:	*
dtype028
6readout/dropout_2/dropout/random_uniform/RandomUniform
(readout/dropout_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2*
(readout/dropout_2/dropout/GreaterEqual/yţ
&readout/dropout_2/dropout/GreaterEqualGreaterEqual?readout/dropout_2/dropout/random_uniform/RandomUniform:output:01readout/dropout_2/dropout/GreaterEqual/y:output:0*
T0*
_output_shapes
:	2(
&readout/dropout_2/dropout/GreaterEqual­
readout/dropout_2/dropout/CastCast*readout/dropout_2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*
_output_shapes
:	2 
readout/dropout_2/dropout/Castş
readout/dropout_2/dropout/Mul_1Mul!readout/dropout_2/dropout/Mul:z:0"readout/dropout_2/dropout/Cast:y:0*
T0*
_output_shapes
:	2!
readout/dropout_2/dropout/Mul_1Á
&readout/dense_14/MatMul/ReadVariableOpReadVariableOp/readout_dense_14_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype02(
&readout/dense_14/MatMul/ReadVariableOpş
readout/dense_14/MatMulMatMul#readout/dropout_2/dropout/Mul_1:z:0.readout/dense_14/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:@2
readout/dense_14/MatMulż
'readout/dense_14/BiasAdd/ReadVariableOpReadVariableOp0readout_dense_14_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02)
'readout/dense_14/BiasAdd/ReadVariableOpź
readout/dense_14/BiasAddBiasAdd!readout/dense_14/MatMul:product:0/readout/dense_14/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:@2
readout/dense_14/BiasAdd
readout/dense_14/TanhTanh!readout/dense_14/BiasAdd:output:0*
T0*
_output_shapes

:@2
readout/dense_14/Tanh
readout/dropout_3/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?2!
readout/dropout_3/dropout/Constł
readout/dropout_3/dropout/MulMulreadout/dense_14/Tanh:y:0(readout/dropout_3/dropout/Const:output:0*
T0*
_output_shapes

:@2
readout/dropout_3/dropout/Mul
readout/dropout_3/dropout/ShapeConst*
_output_shapes
:*
dtype0*
valueB"   @   2!
readout/dropout_3/dropout/Shapeá
6readout/dropout_3/dropout/random_uniform/RandomUniformRandomUniform(readout/dropout_3/dropout/Shape:output:0*
T0*
_output_shapes

:@*
dtype028
6readout/dropout_3/dropout/random_uniform/RandomUniform
(readout/dropout_3/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2*
(readout/dropout_3/dropout/GreaterEqual/yý
&readout/dropout_3/dropout/GreaterEqualGreaterEqual?readout/dropout_3/dropout/random_uniform/RandomUniform:output:01readout/dropout_3/dropout/GreaterEqual/y:output:0*
T0*
_output_shapes

:@2(
&readout/dropout_3/dropout/GreaterEqualŹ
readout/dropout_3/dropout/CastCast*readout/dropout_3/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*
_output_shapes

:@2 
readout/dropout_3/dropout/Castš
readout/dropout_3/dropout/Mul_1Mul!readout/dropout_3/dropout/Mul:z:0"readout/dropout_3/dropout/Cast:y:0*
T0*
_output_shapes

:@2!
readout/dropout_3/dropout/Mul_1Ŕ
&readout/dense_15/MatMul/ReadVariableOpReadVariableOp/readout_dense_15_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02(
&readout/dense_15/MatMul/ReadVariableOpş
readout/dense_15/MatMulMatMul#readout/dropout_3/dropout/Mul_1:z:0.readout/dense_15/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
readout/dense_15/MatMulż
'readout/dense_15/BiasAdd/ReadVariableOpReadVariableOp0readout_dense_15_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'readout/dense_15/BiasAdd/ReadVariableOpź
readout/dense_15/BiasAddBiasAdd!readout/dense_15/MatMul:product:0/readout/dense_15/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:2
readout/dense_15/BiasAddq
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙2
Reshape/shape}
ReshapeReshape!readout/dense_15/BiasAdd:output:0Reshape/shape:output:0*
T0*
_output_shapes
:2	
Reshapeq
IdentityIdentityReshape:output:0^StatefulPartitionedCall*
T0*
_output_shapes
:2

Identity"
identityIdentity:output:0*b
_input_shapesQ
O:˙˙˙˙˙˙˙˙˙::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:J F
#
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_nameinput
Ú
}
(__inference_dense_12_layer_call_fn_55233

inputs
unknown
	unknown_0
identity˘StatefulPartitionedCalló
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_12_layer_call_and_return_conditional_losses_537212
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*.
_input_shapes
:˙˙˙˙˙˙˙˙˙@::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙@
 
_user_specified_nameinputs
ź
¨
.__inference_create_message_layer_call_fn_53625
dense_8_input
unknown
	unknown_0
	unknown_1
	unknown_2
identity˘StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCalldense_8_inputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_create_message_layer_call_and_return_conditional_losses_536142
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:˙˙˙˙˙˙˙˙˙ ::::22
StatefulPartitionedCallStatefulPartitionedCall:V R
'
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
'
_user_specified_namedense_8_input
Ů
¸
'__inference_readout_layer_call_fn_55116

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity˘StatefulPartitionedCallŚ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_readout_layer_call_and_return_conditional_losses_540202
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:˙˙˙˙˙˙˙˙˙@::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙@
 
_user_specified_nameinputs
Â
Ą
A__inference_critic_layer_call_and_return_conditional_losses_54300	
input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
readout_54284
readout_54286
readout_54288
readout_54290
readout_54292
readout_54294
identity˘StatefulPartitionedCall˘readout/StatefulPartitionedCallŻ
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:J*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8 **
f%R#
!__inference_message_passing_424792
StatefulPartitionedCallś
PartitionedCallPartitionedCall StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *0
f+R)
'__inference_generate_readout_input_15772
PartitionedCallŮ
readout/StatefulPartitionedCallStatefulPartitionedCallPartitionedCall:output:0readout_54284readout_54286readout_54288readout_54290readout_54292readout_54294*
Tin
	2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_readout_layer_call_and_return_conditional_losses_541642!
readout/StatefulPartitionedCallq
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙2
Reshape/shape
ReshapeReshape(readout/StatefulPartitionedCall:output:0Reshape/shape:output:0*
T0*
_output_shapes
:2	
Reshape
IdentityIdentityReshape:output:0^StatefulPartitionedCall ^readout/StatefulPartitionedCall*
T0*
_output_shapes
:2

Identity"
identityIdentity:output:0*b
_input_shapesQ
O:˙˙˙˙˙˙˙˙˙::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall2B
readout/StatefulPartitionedCallreadout/StatefulPartitionedCall:J F
#
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_nameinput
ˇ
Ć
F__inference_link_update_layer_call_and_return_conditional_losses_53779

inputs
dense_10_53763
dense_10_53765
dense_11_53768
dense_11_53770
dense_12_53773
dense_12_53775
identity˘ dense_10/StatefulPartitionedCall˘ dense_11/StatefulPartitionedCall˘ dense_12/StatefulPartitionedCall
 dense_10/StatefulPartitionedCallStatefulPartitionedCallinputsdense_10_53763dense_10_53765*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_10_layer_call_and_return_conditional_losses_536672"
 dense_10/StatefulPartitionedCall´
 dense_11/StatefulPartitionedCallStatefulPartitionedCall)dense_10/StatefulPartitionedCall:output:0dense_11_53768dense_11_53770*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_11_layer_call_and_return_conditional_losses_536942"
 dense_11/StatefulPartitionedCall´
 dense_12/StatefulPartitionedCallStatefulPartitionedCall)dense_11/StatefulPartitionedCall:output:0dense_12_53773dense_12_53775*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_12_layer_call_and_return_conditional_losses_537212"
 dense_12/StatefulPartitionedCallć
IdentityIdentity)dense_12/StatefulPartitionedCall:output:0!^dense_10/StatefulPartitionedCall!^dense_11/StatefulPartitionedCall!^dense_12/StatefulPartitionedCall*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:˙˙˙˙˙˙˙˙˙0::::::2D
 dense_10/StatefulPartitionedCall dense_10/StatefulPartitionedCall2D
 dense_11/StatefulPartitionedCall dense_11/StatefulPartitionedCall2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall:O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙0
 
_user_specified_nameinputs

Ş
B__inference_dense_8_layer_call_and_return_conditional_losses_53539

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: @*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@2	
BiasAddX
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@2
Tanh\
IdentityIdentityTanh:y:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@2

Identity"
identityIdentity:output:0*.
_input_shapes
:˙˙˙˙˙˙˙˙˙ :::O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
 
_user_specified_nameinputs
ś
Ę
B__inference_readout_layer_call_and_return_conditional_losses_53996
dense_13_input
dense_13_53978
dense_13_53980
dense_14_53984
dense_14_53986
dense_15_53990
dense_15_53992
identity˘ dense_13/StatefulPartitionedCall˘ dense_14/StatefulPartitionedCall˘ dense_15/StatefulPartitionedCall
 dense_13/StatefulPartitionedCallStatefulPartitionedCalldense_13_inputdense_13_53978dense_13_53980*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_13_layer_call_and_return_conditional_losses_538452"
 dense_13/StatefulPartitionedCallú
dropout_2/PartitionedCallPartitionedCall)dense_13/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_2_layer_call_and_return_conditional_losses_538782
dropout_2/PartitionedCall­
 dense_14/StatefulPartitionedCallStatefulPartitionedCall"dropout_2/PartitionedCall:output:0dense_14_53984dense_14_53986*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_14_layer_call_and_return_conditional_losses_539022"
 dense_14/StatefulPartitionedCallů
dropout_3/PartitionedCallPartitionedCall)dense_14/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_3_layer_call_and_return_conditional_losses_539352
dropout_3/PartitionedCall­
 dense_15/StatefulPartitionedCallStatefulPartitionedCall"dropout_3/PartitionedCall:output:0dense_15_53990dense_15_53992*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_15_layer_call_and_return_conditional_losses_539582"
 dense_15/StatefulPartitionedCallć
IdentityIdentity)dense_15/StatefulPartitionedCall:output:0!^dense_13/StatefulPartitionedCall!^dense_14/StatefulPartitionedCall!^dense_15/StatefulPartitionedCall*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:˙˙˙˙˙˙˙˙˙@::::::2D
 dense_13/StatefulPartitionedCall dense_13/StatefulPartitionedCall2D
 dense_14/StatefulPartitionedCall dense_14/StatefulPartitionedCall2D
 dense_15/StatefulPartitionedCall dense_15/StatefulPartitionedCall:W S
'
_output_shapes
:˙˙˙˙˙˙˙˙˙@
(
_user_specified_namedense_13_input
Ú
ň
B__inference_readout_layer_call_and_return_conditional_losses_54999

inputs+
'dense_13_matmul_readvariableop_resource,
(dense_13_biasadd_readvariableop_resource+
'dense_14_matmul_readvariableop_resource,
(dense_14_biasadd_readvariableop_resource+
'dense_15_matmul_readvariableop_resource,
(dense_15_biasadd_readvariableop_resource
identityŠ
dense_13/MatMul/ReadVariableOpReadVariableOp'dense_13_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype02 
dense_13/MatMul/ReadVariableOp
dense_13/MatMulMatMulinputs&dense_13/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
dense_13/MatMul¨
dense_13/BiasAdd/ReadVariableOpReadVariableOp(dense_13_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02!
dense_13/BiasAdd/ReadVariableOp
dense_13/BiasAddBiasAdddense_13/MatMul:product:0'dense_13/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
dense_13/BiasAddk
dense_13/TanhTanhdense_13/BiasAdd:output:0*
T0*
_output_shapes
:	2
dense_13/Tanhq
dropout_2/IdentityIdentitydense_13/Tanh:y:0*
T0*
_output_shapes
:	2
dropout_2/IdentityŠ
dense_14/MatMul/ReadVariableOpReadVariableOp'dense_14_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype02 
dense_14/MatMul/ReadVariableOp
dense_14/MatMulMatMuldropout_2/Identity:output:0&dense_14/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:@2
dense_14/MatMul§
dense_14/BiasAdd/ReadVariableOpReadVariableOp(dense_14_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
dense_14/BiasAdd/ReadVariableOp
dense_14/BiasAddBiasAdddense_14/MatMul:product:0'dense_14/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:@2
dense_14/BiasAddj
dense_14/TanhTanhdense_14/BiasAdd:output:0*
T0*
_output_shapes

:@2
dense_14/Tanhp
dropout_3/IdentityIdentitydense_14/Tanh:y:0*
T0*
_output_shapes

:@2
dropout_3/Identity¨
dense_15/MatMul/ReadVariableOpReadVariableOp'dense_15_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02 
dense_15/MatMul/ReadVariableOp
dense_15/MatMulMatMuldropout_3/Identity:output:0&dense_15/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
dense_15/MatMul§
dense_15/BiasAdd/ReadVariableOpReadVariableOp(dense_15_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_15/BiasAdd/ReadVariableOp
dense_15/BiasAddBiasAdddense_15/MatMul:product:0'dense_15/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:2
dense_15/BiasAddd
IdentityIdentitydense_15/BiasAdd:output:0*
T0*
_output_shapes

:2

Identity"
identityIdentity:output:0*5
_input_shapes$
":@:::::::F B

_output_shapes

:@
 
_user_specified_nameinputs
ú)
ň
B__inference_readout_layer_call_and_return_conditional_losses_54138

inputs+
'dense_13_matmul_readvariableop_resource,
(dense_13_biasadd_readvariableop_resource+
'dense_14_matmul_readvariableop_resource,
(dense_14_biasadd_readvariableop_resource+
'dense_15_matmul_readvariableop_resource,
(dense_15_biasadd_readvariableop_resource
identityŠ
dense_13/MatMul/ReadVariableOpReadVariableOp'dense_13_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype02 
dense_13/MatMul/ReadVariableOp
dense_13/MatMulMatMulinputs&dense_13/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
dense_13/MatMul¨
dense_13/BiasAdd/ReadVariableOpReadVariableOp(dense_13_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02!
dense_13/BiasAdd/ReadVariableOp
dense_13/BiasAddBiasAdddense_13/MatMul:product:0'dense_13/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
dense_13/BiasAddk
dense_13/TanhTanhdense_13/BiasAdd:output:0*
T0*
_output_shapes
:	2
dense_13/Tanhw
dropout_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?2
dropout_2/dropout/Const
dropout_2/dropout/MulMuldense_13/Tanh:y:0 dropout_2/dropout/Const:output:0*
T0*
_output_shapes
:	2
dropout_2/dropout/Mul
dropout_2/dropout/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2
dropout_2/dropout/ShapeĘ
.dropout_2/dropout/random_uniform/RandomUniformRandomUniform dropout_2/dropout/Shape:output:0*
T0*
_output_shapes
:	*
dtype020
.dropout_2/dropout/random_uniform/RandomUniform
 dropout_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2"
 dropout_2/dropout/GreaterEqual/yŢ
dropout_2/dropout/GreaterEqualGreaterEqual7dropout_2/dropout/random_uniform/RandomUniform:output:0)dropout_2/dropout/GreaterEqual/y:output:0*
T0*
_output_shapes
:	2 
dropout_2/dropout/GreaterEqual
dropout_2/dropout/CastCast"dropout_2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*
_output_shapes
:	2
dropout_2/dropout/Cast
dropout_2/dropout/Mul_1Muldropout_2/dropout/Mul:z:0dropout_2/dropout/Cast:y:0*
T0*
_output_shapes
:	2
dropout_2/dropout/Mul_1Š
dense_14/MatMul/ReadVariableOpReadVariableOp'dense_14_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype02 
dense_14/MatMul/ReadVariableOp
dense_14/MatMulMatMuldropout_2/dropout/Mul_1:z:0&dense_14/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:@2
dense_14/MatMul§
dense_14/BiasAdd/ReadVariableOpReadVariableOp(dense_14_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
dense_14/BiasAdd/ReadVariableOp
dense_14/BiasAddBiasAdddense_14/MatMul:product:0'dense_14/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:@2
dense_14/BiasAddj
dense_14/TanhTanhdense_14/BiasAdd:output:0*
T0*
_output_shapes

:@2
dense_14/Tanhw
dropout_3/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?2
dropout_3/dropout/Const
dropout_3/dropout/MulMuldense_14/Tanh:y:0 dropout_3/dropout/Const:output:0*
T0*
_output_shapes

:@2
dropout_3/dropout/Mul
dropout_3/dropout/ShapeConst*
_output_shapes
:*
dtype0*
valueB"   @   2
dropout_3/dropout/ShapeÉ
.dropout_3/dropout/random_uniform/RandomUniformRandomUniform dropout_3/dropout/Shape:output:0*
T0*
_output_shapes

:@*
dtype020
.dropout_3/dropout/random_uniform/RandomUniform
 dropout_3/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2"
 dropout_3/dropout/GreaterEqual/yÝ
dropout_3/dropout/GreaterEqualGreaterEqual7dropout_3/dropout/random_uniform/RandomUniform:output:0)dropout_3/dropout/GreaterEqual/y:output:0*
T0*
_output_shapes

:@2 
dropout_3/dropout/GreaterEqual
dropout_3/dropout/CastCast"dropout_3/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*
_output_shapes

:@2
dropout_3/dropout/Cast
dropout_3/dropout/Mul_1Muldropout_3/dropout/Mul:z:0dropout_3/dropout/Cast:y:0*
T0*
_output_shapes

:@2
dropout_3/dropout/Mul_1¨
dense_15/MatMul/ReadVariableOpReadVariableOp'dense_15_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02 
dense_15/MatMul/ReadVariableOp
dense_15/MatMulMatMuldropout_3/dropout/Mul_1:z:0&dense_15/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
dense_15/MatMul§
dense_15/BiasAdd/ReadVariableOpReadVariableOp(dense_15_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_15/BiasAdd/ReadVariableOp
dense_15/BiasAddBiasAdddense_15/MatMul:product:0'dense_15/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:2
dense_15/BiasAddd
IdentityIdentitydense_15/BiasAdd:output:0*
T0*
_output_shapes

:2

Identity"
identityIdentity:output:0*5
_input_shapes$
":@:::::::F B

_output_shapes

:@
 
_user_specified_nameinputs
Ü
}
(__inference_dense_10_layer_call_fn_55193

inputs
unknown
	unknown_0
identity˘StatefulPartitionedCallô
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_10_layer_call_and_return_conditional_losses_536672
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*.
_input_shapes
:˙˙˙˙˙˙˙˙˙0::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙0
 
_user_specified_nameinputs

E
)__inference_dropout_2_layer_call_fn_55280

inputs
identityĂ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_2_layer_call_and_return_conditional_losses_538782
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*'
_input_shapes
:˙˙˙˙˙˙˙˙˙:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
˘

B__inference_readout_layer_call_and_return_conditional_losses_54020

inputs
dense_13_54002
dense_13_54004
dense_14_54008
dense_14_54010
dense_15_54014
dense_15_54016
identity˘ dense_13/StatefulPartitionedCall˘ dense_14/StatefulPartitionedCall˘ dense_15/StatefulPartitionedCall˘!dropout_2/StatefulPartitionedCall˘!dropout_3/StatefulPartitionedCall
 dense_13/StatefulPartitionedCallStatefulPartitionedCallinputsdense_13_54002dense_13_54004*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_13_layer_call_and_return_conditional_losses_538452"
 dense_13/StatefulPartitionedCall
!dropout_2/StatefulPartitionedCallStatefulPartitionedCall)dense_13/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_2_layer_call_and_return_conditional_losses_538732#
!dropout_2/StatefulPartitionedCallľ
 dense_14/StatefulPartitionedCallStatefulPartitionedCall*dropout_2/StatefulPartitionedCall:output:0dense_14_54008dense_14_54010*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_14_layer_call_and_return_conditional_losses_539022"
 dense_14/StatefulPartitionedCallľ
!dropout_3/StatefulPartitionedCallStatefulPartitionedCall)dense_14/StatefulPartitionedCall:output:0"^dropout_2/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_3_layer_call_and_return_conditional_losses_539302#
!dropout_3/StatefulPartitionedCallľ
 dense_15/StatefulPartitionedCallStatefulPartitionedCall*dropout_3/StatefulPartitionedCall:output:0dense_15_54014dense_15_54016*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_15_layer_call_and_return_conditional_losses_539582"
 dense_15/StatefulPartitionedCallŽ
IdentityIdentity)dense_15/StatefulPartitionedCall:output:0!^dense_13/StatefulPartitionedCall!^dense_14/StatefulPartitionedCall!^dense_15/StatefulPartitionedCall"^dropout_2/StatefulPartitionedCall"^dropout_3/StatefulPartitionedCall*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:˙˙˙˙˙˙˙˙˙@::::::2D
 dense_13/StatefulPartitionedCall dense_13/StatefulPartitionedCall2D
 dense_14/StatefulPartitionedCall dense_14/StatefulPartitionedCall2D
 dense_15/StatefulPartitionedCall dense_15/StatefulPartitionedCall2F
!dropout_2/StatefulPartitionedCall!dropout_2/StatefulPartitionedCall2F
!dropout_3/StatefulPartitionedCall!dropout_3/StatefulPartitionedCall:O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙@
 
_user_specified_nameinputs
Ü
}
(__inference_dense_13_layer_call_fn_55253

inputs
unknown
	unknown_0
identity˘StatefulPartitionedCallô
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_13_layer_call_and_return_conditional_losses_538452
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*.
_input_shapes
:˙˙˙˙˙˙˙˙˙@::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙@
 
_user_specified_nameinputs


Ó
&__inference_critic_layer_call_fn_54599
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14
identity˘StatefulPartitionedCall 
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_critic_layer_call_and_return_conditional_losses_543002
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes
:2

Identity"
identityIdentity:output:0*b
_input_shapesQ
O:˙˙˙˙˙˙˙˙˙::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:L H
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
!
_user_specified_name	input_1

b
)__inference_dropout_3_layer_call_fn_55322

inputs
identity˘StatefulPartitionedCallÚ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_3_layer_call_and_return_conditional_losses_539302
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@2

Identity"
identityIdentity:output:0*&
_input_shapes
:˙˙˙˙˙˙˙˙˙@22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙@
 
_user_specified_nameinputs
ľ
¸
'__inference_readout_layer_call_fn_55016

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity˘StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_readout_layer_call_and_return_conditional_losses_541382
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes

:2

Identity"
identityIdentity:output:0*5
_input_shapes$
":@::::::22
StatefulPartitionedCallStatefulPartitionedCall:F B

_output_shapes

:@
 
_user_specified_nameinputs

Ş
B__inference_dense_9_layer_call_and_return_conditional_losses_53566

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2	
BiasAddX
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2
Tanh\
IdentityIdentityTanh:y:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*.
_input_shapes
:˙˙˙˙˙˙˙˙˙@:::O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙@
 
_user_specified_nameinputs
á
ź
+__inference_link_update_layer_call_fn_54933

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity˘StatefulPartitionedCallŞ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_link_update_layer_call_and_return_conditional_losses_538152
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:˙˙˙˙˙˙˙˙˙0::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙0
 
_user_specified_nameinputs

Ş
B__inference_dense_9_layer_call_and_return_conditional_losses_55164

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2	
BiasAddX
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2
Tanh\
IdentityIdentityTanh:y:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*.
_input_shapes
:˙˙˙˙˙˙˙˙˙@:::O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙@
 
_user_specified_nameinputs
Ą
Ť
C__inference_dense_14_layer_call_and_return_conditional_losses_53902

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@2	
BiasAddX
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@2
Tanh\
IdentityIdentityTanh:y:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@2

Identity"
identityIdentity:output:0*/
_input_shapes
:˙˙˙˙˙˙˙˙˙:::P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
ź
¨
.__inference_create_message_layer_call_fn_53652
dense_8_input
unknown
	unknown_0
	unknown_1
	unknown_2
identity˘StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCalldense_8_inputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_create_message_layer_call_and_return_conditional_losses_536412
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:˙˙˙˙˙˙˙˙˙ ::::22
StatefulPartitionedCallStatefulPartitionedCall:V R
'
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
'
_user_specified_namedense_8_input

Â
B__inference_readout_layer_call_and_return_conditional_losses_54058

inputs
dense_13_54040
dense_13_54042
dense_14_54046
dense_14_54048
dense_15_54052
dense_15_54054
identity˘ dense_13/StatefulPartitionedCall˘ dense_14/StatefulPartitionedCall˘ dense_15/StatefulPartitionedCall
 dense_13/StatefulPartitionedCallStatefulPartitionedCallinputsdense_13_54040dense_13_54042*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_13_layer_call_and_return_conditional_losses_538452"
 dense_13/StatefulPartitionedCallú
dropout_2/PartitionedCallPartitionedCall)dense_13/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_2_layer_call_and_return_conditional_losses_538782
dropout_2/PartitionedCall­
 dense_14/StatefulPartitionedCallStatefulPartitionedCall"dropout_2/PartitionedCall:output:0dense_14_54046dense_14_54048*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_14_layer_call_and_return_conditional_losses_539022"
 dense_14/StatefulPartitionedCallů
dropout_3/PartitionedCallPartitionedCall)dense_14/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_3_layer_call_and_return_conditional_losses_539352
dropout_3/PartitionedCall­
 dense_15/StatefulPartitionedCallStatefulPartitionedCall"dropout_3/PartitionedCall:output:0dense_15_54052dense_15_54054*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_15_layer_call_and_return_conditional_losses_539582"
 dense_15/StatefulPartitionedCallć
IdentityIdentity)dense_15/StatefulPartitionedCall:output:0!^dense_13/StatefulPartitionedCall!^dense_14/StatefulPartitionedCall!^dense_15/StatefulPartitionedCall*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:˙˙˙˙˙˙˙˙˙@::::::2D
 dense_13/StatefulPartitionedCall dense_13/StatefulPartitionedCall2D
 dense_14/StatefulPartitionedCall dense_14/StatefulPartitionedCall2D
 dense_15/StatefulPartitionedCall dense_15/StatefulPartitionedCall:O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙@
 
_user_specified_nameinputs
%
Ł
__inference_call_43925	
input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_83
/readout_dense_13_matmul_readvariableop_resource4
0readout_dense_13_biasadd_readvariableop_resource3
/readout_dense_14_matmul_readvariableop_resource4
0readout_dense_14_biasadd_readvariableop_resource3
/readout_dense_15_matmul_readvariableop_resource4
0readout_dense_15_biasadd_readvariableop_resource
identity˘StatefulPartitionedCallŻ
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:J*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8 **
f%R#
!__inference_message_passing_424792
StatefulPartitionedCallś
PartitionedCallPartitionedCall StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *0
f+R)
'__inference_generate_readout_input_15772
PartitionedCallÁ
&readout/dense_13/MatMul/ReadVariableOpReadVariableOp/readout_dense_13_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype02(
&readout/dense_13/MatMul/ReadVariableOp°
readout/dense_13/MatMulMatMulPartitionedCall:output:0.readout/dense_13/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
readout/dense_13/MatMulŔ
'readout/dense_13/BiasAdd/ReadVariableOpReadVariableOp0readout_dense_13_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02)
'readout/dense_13/BiasAdd/ReadVariableOp˝
readout/dense_13/BiasAddBiasAdd!readout/dense_13/MatMul:product:0/readout/dense_13/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
readout/dense_13/BiasAdd
readout/dense_13/TanhTanh!readout/dense_13/BiasAdd:output:0*
T0*
_output_shapes
:	2
readout/dense_13/Tanh
readout/dropout_2/IdentityIdentityreadout/dense_13/Tanh:y:0*
T0*
_output_shapes
:	2
readout/dropout_2/IdentityÁ
&readout/dense_14/MatMul/ReadVariableOpReadVariableOp/readout_dense_14_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype02(
&readout/dense_14/MatMul/ReadVariableOpş
readout/dense_14/MatMulMatMul#readout/dropout_2/Identity:output:0.readout/dense_14/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:@2
readout/dense_14/MatMulż
'readout/dense_14/BiasAdd/ReadVariableOpReadVariableOp0readout_dense_14_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02)
'readout/dense_14/BiasAdd/ReadVariableOpź
readout/dense_14/BiasAddBiasAdd!readout/dense_14/MatMul:product:0/readout/dense_14/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:@2
readout/dense_14/BiasAdd
readout/dense_14/TanhTanh!readout/dense_14/BiasAdd:output:0*
T0*
_output_shapes

:@2
readout/dense_14/Tanh
readout/dropout_3/IdentityIdentityreadout/dense_14/Tanh:y:0*
T0*
_output_shapes

:@2
readout/dropout_3/IdentityŔ
&readout/dense_15/MatMul/ReadVariableOpReadVariableOp/readout_dense_15_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02(
&readout/dense_15/MatMul/ReadVariableOpş
readout/dense_15/MatMulMatMul#readout/dropout_3/Identity:output:0.readout/dense_15/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
readout/dense_15/MatMulż
'readout/dense_15/BiasAdd/ReadVariableOpReadVariableOp0readout_dense_15_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'readout/dense_15/BiasAdd/ReadVariableOpź
readout/dense_15/BiasAddBiasAdd!readout/dense_15/MatMul:product:0/readout/dense_15/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:2
readout/dense_15/BiasAddq
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙2
Reshape/shape}
ReshapeReshape!readout/dense_15/BiasAdd:output:0Reshape/shape:output:0*
T0*
_output_shapes
:2	
Reshapeq
IdentityIdentityReshape:output:0^StatefulPartitionedCall*
T0*
_output_shapes
:2

Identity"
identityIdentity:output:0*b
_input_shapesQ
O:˙˙˙˙˙˙˙˙˙::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:J F
#
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_nameinput

Ť
C__inference_dense_12_layer_call_and_return_conditional_losses_53721

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2	
BiasAddX
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2
Tanh\
IdentityIdentityTanh:y:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*.
_input_shapes
:˙˙˙˙˙˙˙˙˙@:::O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙@
 
_user_specified_nameinputs
Ţ	
Đ
#__inference_signature_wrapper_54411
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14
identity˘StatefulPartitionedCall˙
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *)
f$R"
 __inference__wrapped_model_535242
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes
:2

Identity"
identityIdentity:output:0*b
_input_shapesQ
O:˙˙˙˙˙˙˙˙˙::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:L H
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
!
_user_specified_name	input_1
ü	
Ń
&__inference_critic_layer_call_fn_54750	
input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14
identity˘StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_critic_layer_call_and_return_conditional_losses_543002
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes
:2

Identity"
identityIdentity:output:0*b
_input_shapesQ
O:˙˙˙˙˙˙˙˙˙::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:J F
#
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_nameinput
ź°

!__inference_message_passing_44266	
input9
5create_message_dense_8_matmul_readvariableop_resource:
6create_message_dense_8_biasadd_readvariableop_resource9
5create_message_dense_9_matmul_readvariableop_resource:
6create_message_dense_9_biasadd_readvariableop_resource7
3link_update_dense_10_matmul_readvariableop_resource8
4link_update_dense_10_biasadd_readvariableop_resource7
3link_update_dense_11_matmul_readvariableop_resource8
4link_update_dense_11_biasadd_readvariableop_resource7
3link_update_dense_12_matmul_readvariableop_resource8
4link_update_dense_12_biasadd_readvariableop_resource
identityo
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"   J   2
Reshape/shapee
ReshapeReshapeinputReshape/shape:output:0*
T0*
_output_shapes

:J2	
Reshapeq
transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose/permw
	transpose	TransposeReshape:output:0transpose/perm:output:0*
T0*
_output_shapes

:J2
	transpose}
Pad/paddingsConst*
_output_shapes

:*
dtype0*)
value B"               2
Pad/paddings`
PadPadtranspose:y:0Pad/paddings:output:0*
T0*
_output_shapes

:J2
Pad	
GatherV2/indicesConst*
_output_shapes	
:*
dtype0*ą
value§B¤"                                                      
   
   
   
   
                                                                                                                                            !   !   !   !   !   !   "   "   "   "   "   "   "   "   #   #   #   %   %   %   %   %   %   &   &   &   '   '   '   '   '   '   '   '   (   (   (   (   )   )   )   )   )   )   )   )   *   *   +   +   +   +   +   +   ,   ,   ,   ,   -   -   -   -   -   -   -   -   .   .   .   .   /   /   /   0   0   0   0   0   1   1   1   1   1   1   1   1   2   2   2   2   3   3   3   3   4   4   4   4   4   5   5   5   6   6   6   6   7   7   7   8   8   8   8   8   8   9   9   9   9   9   9   9   9   :   :   :   :   ;   ;   ;   ;   <   <   =   =   =   =   >   >   >   >   ?   ?   ?   ?   ?   @   @   A   A   A   A   A   A   B   B   B   B   C   C   C   C   C   C   C   C   D   D   D   D   E   E   E   E   F   F   F   F   G   G   G   G   H   H   H   H   H   I   I   I   I   I   I   I   I   2
GatherV2/indices`
GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2/axisŽ
GatherV2GatherV2Pad:output:0GatherV2/indices:output:0GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:	2

GatherV2	
GatherV2_1/indicesConst*
_output_shapes	
:*
dtype0*ą
value§B¤"      
                                  #   %   &   '   5   6   7   8   9          D   E   F   G   H   I                                                                         !   "                                 B   C   D   E   F   G   H   I   :   ;   <   =   >   ?   @   A         
   D   E   F   G   H   I   2   3   4   :   ;   <   =   >   ?   @   A   *   +   ,   -   :   ;   <   =   >   ?   @   A   (   )   D   E   F   G   H   I   .   /   0   1   :   ;   <   =   >   ?   @   A   *   +   ,   -   2   3   4   5   6   7   8   9   :   ;   <   =   >   ?   @   A   #   %   &   '   .   /   0   1   5   6   7   8   9         
   .   /   0   1   2   3   4   D   E   F   G   H   I   :   ;   <   =   >   ?   @   A          !   "   #   %   &   '   (   )   *   +   ,   -   .   /   0   1   5   6   7   8   9   B   C   D   E   F   G   H   I          !   "   :   ;   <   =   >   ?   @   A                      !   "   #   %   &   '   *   +   ,   -   5   6   7   8   9   :   ;   <   =   >   ?   @   A   2
GatherV2_1/indicesd
GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2_1/axisś

GatherV2_1GatherV2Pad:output:0GatherV2_1/indices:output:0GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:	2

GatherV2_1\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis
concatConcatV2GatherV2:output:0GatherV2_1:output:0concat/axis:output:0*
N*
T0*
_output_shapes
:	 2
concatŇ
,create_message/dense_8/MatMul/ReadVariableOpReadVariableOp5create_message_dense_8_matmul_readvariableop_resource*
_output_shapes

: @*
dtype02.
,create_message/dense_8/MatMul/ReadVariableOpš
create_message/dense_8/MatMulMatMulconcat:output:04create_message/dense_8/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	@2
create_message/dense_8/MatMulŃ
-create_message/dense_8/BiasAdd/ReadVariableOpReadVariableOp6create_message_dense_8_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-create_message/dense_8/BiasAdd/ReadVariableOpŐ
create_message/dense_8/BiasAddBiasAdd'create_message/dense_8/MatMul:product:05create_message/dense_8/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	@2 
create_message/dense_8/BiasAdd
create_message/dense_8/TanhTanh'create_message/dense_8/BiasAdd:output:0*
T0*
_output_shapes
:	@2
create_message/dense_8/TanhŇ
,create_message/dense_9/MatMul/ReadVariableOpReadVariableOp5create_message_dense_9_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02.
,create_message/dense_9/MatMul/ReadVariableOpÉ
create_message/dense_9/MatMulMatMulcreate_message/dense_8/Tanh:y:04create_message/dense_9/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
create_message/dense_9/MatMulŃ
-create_message/dense_9/BiasAdd/ReadVariableOpReadVariableOp6create_message_dense_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-create_message/dense_9/BiasAdd/ReadVariableOpŐ
create_message/dense_9/BiasAddBiasAdd'create_message/dense_9/MatMul:product:05create_message/dense_9/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	2 
create_message/dense_9/BiasAdd
create_message/dense_9/TanhTanh'create_message/dense_9/BiasAdd:output:0*
T0*
_output_shapes
:	2
create_message/dense_9/Tanh˛
PartitionedCallPartitionedCallcreate_message/dense_9/Tanh:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:J * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *-
f(R&
$__inference_message_aggregation_12562
PartitionedCall`
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat_1/axis
concat_1ConcatV2Pad:output:0PartitionedCall:output:0concat_1/axis:output:0*
N*
T0*
_output_shapes

:J02

concat_1Í
*link_update/dense_10/MatMul/ReadVariableOpReadVariableOp3link_update_dense_10_matmul_readvariableop_resource*
_output_shapes
:	0*
dtype02,
*link_update/dense_10/MatMul/ReadVariableOpľ
link_update/dense_10/MatMulMatMulconcat_1:output:02link_update/dense_10/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	J2
link_update/dense_10/MatMulĚ
+link_update/dense_10/BiasAdd/ReadVariableOpReadVariableOp4link_update_dense_10_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02-
+link_update/dense_10/BiasAdd/ReadVariableOpÍ
link_update/dense_10/BiasAddBiasAdd%link_update/dense_10/MatMul:product:03link_update/dense_10/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	J2
link_update/dense_10/BiasAdd
link_update/dense_10/TanhTanh%link_update/dense_10/BiasAdd:output:0*
T0*
_output_shapes
:	J2
link_update/dense_10/TanhÍ
*link_update/dense_11/MatMul/ReadVariableOpReadVariableOp3link_update_dense_11_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype02,
*link_update/dense_11/MatMul/ReadVariableOpŔ
link_update/dense_11/MatMulMatMullink_update/dense_10/Tanh:y:02link_update/dense_11/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:J@2
link_update/dense_11/MatMulË
+link_update/dense_11/BiasAdd/ReadVariableOpReadVariableOp4link_update_dense_11_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02-
+link_update/dense_11/BiasAdd/ReadVariableOpĚ
link_update/dense_11/BiasAddBiasAdd%link_update/dense_11/MatMul:product:03link_update/dense_11/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:J@2
link_update/dense_11/BiasAdd
link_update/dense_11/TanhTanh%link_update/dense_11/BiasAdd:output:0*
T0*
_output_shapes

:J@2
link_update/dense_11/TanhĚ
*link_update/dense_12/MatMul/ReadVariableOpReadVariableOp3link_update_dense_12_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02,
*link_update/dense_12/MatMul/ReadVariableOpŔ
link_update/dense_12/MatMulMatMullink_update/dense_11/Tanh:y:02link_update/dense_12/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:J2
link_update/dense_12/MatMulË
+link_update/dense_12/BiasAdd/ReadVariableOpReadVariableOp4link_update_dense_12_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02-
+link_update/dense_12/BiasAdd/ReadVariableOpĚ
link_update/dense_12/BiasAddBiasAdd%link_update/dense_12/MatMul:product:03link_update/dense_12/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:J2
link_update/dense_12/BiasAdd
link_update/dense_12/TanhTanh%link_update/dense_12/BiasAdd:output:0*
T0*
_output_shapes

:J2
link_update/dense_12/Tanh	
GatherV2_2/indicesConst*
_output_shapes	
:*
dtype0*ą
value§B¤"                                                      
   
   
   
   
                                                                                                                                            !   !   !   !   !   !   "   "   "   "   "   "   "   "   #   #   #   %   %   %   %   %   %   &   &   &   '   '   '   '   '   '   '   '   (   (   (   (   )   )   )   )   )   )   )   )   *   *   +   +   +   +   +   +   ,   ,   ,   ,   -   -   -   -   -   -   -   -   .   .   .   .   /   /   /   0   0   0   0   0   1   1   1   1   1   1   1   1   2   2   2   2   3   3   3   3   4   4   4   4   4   5   5   5   6   6   6   6   7   7   7   8   8   8   8   8   8   9   9   9   9   9   9   9   9   :   :   :   :   ;   ;   ;   ;   <   <   =   =   =   =   >   >   >   >   ?   ?   ?   ?   ?   @   @   A   A   A   A   A   A   B   B   B   B   C   C   C   C   C   C   C   C   D   D   D   D   E   E   E   E   F   F   F   F   G   G   G   G   H   H   H   H   H   I   I   I   I   I   I   I   I   2
GatherV2_2/indicesd
GatherV2_2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2_2/axisÇ

GatherV2_2GatherV2link_update/dense_12/Tanh:y:0GatherV2_2/indices:output:0GatherV2_2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:	2

GatherV2_2	
GatherV2_3/indicesConst*
_output_shapes	
:*
dtype0*ą
value§B¤"      
                                  #   %   &   '   5   6   7   8   9          D   E   F   G   H   I                                                                         !   "                                 B   C   D   E   F   G   H   I   :   ;   <   =   >   ?   @   A         
   D   E   F   G   H   I   2   3   4   :   ;   <   =   >   ?   @   A   *   +   ,   -   :   ;   <   =   >   ?   @   A   (   )   D   E   F   G   H   I   .   /   0   1   :   ;   <   =   >   ?   @   A   *   +   ,   -   2   3   4   5   6   7   8   9   :   ;   <   =   >   ?   @   A   #   %   &   '   .   /   0   1   5   6   7   8   9         
   .   /   0   1   2   3   4   D   E   F   G   H   I   :   ;   <   =   >   ?   @   A          !   "   #   %   &   '   (   )   *   +   ,   -   .   /   0   1   5   6   7   8   9   B   C   D   E   F   G   H   I          !   "   :   ;   <   =   >   ?   @   A                      !   "   #   %   &   '   *   +   ,   -   5   6   7   8   9   :   ;   <   =   >   ?   @   A   2
GatherV2_3/indicesd
GatherV2_3/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2_3/axisÇ

GatherV2_3GatherV2link_update/dense_12/Tanh:y:0GatherV2_3/indices:output:0GatherV2_3/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:	2

GatherV2_3`
concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat_2/axis
concat_2ConcatV2GatherV2_2:output:0GatherV2_3:output:0concat_2/axis:output:0*
N*
T0*
_output_shapes
:	 2

concat_2Ö
.create_message/dense_8/MatMul_1/ReadVariableOpReadVariableOp5create_message_dense_8_matmul_readvariableop_resource*
_output_shapes

: @*
dtype020
.create_message/dense_8/MatMul_1/ReadVariableOpÁ
create_message/dense_8/MatMul_1MatMulconcat_2:output:06create_message/dense_8/MatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	@2!
create_message/dense_8/MatMul_1Ő
/create_message/dense_8/BiasAdd_1/ReadVariableOpReadVariableOp6create_message_dense_8_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype021
/create_message/dense_8/BiasAdd_1/ReadVariableOpÝ
 create_message/dense_8/BiasAdd_1BiasAdd)create_message/dense_8/MatMul_1:product:07create_message/dense_8/BiasAdd_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	@2"
 create_message/dense_8/BiasAdd_1
create_message/dense_8/Tanh_1Tanh)create_message/dense_8/BiasAdd_1:output:0*
T0*
_output_shapes
:	@2
create_message/dense_8/Tanh_1Ö
.create_message/dense_9/MatMul_1/ReadVariableOpReadVariableOp5create_message_dense_9_matmul_readvariableop_resource*
_output_shapes

:@*
dtype020
.create_message/dense_9/MatMul_1/ReadVariableOpŃ
create_message/dense_9/MatMul_1MatMul!create_message/dense_8/Tanh_1:y:06create_message/dense_9/MatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	2!
create_message/dense_9/MatMul_1Ő
/create_message/dense_9/BiasAdd_1/ReadVariableOpReadVariableOp6create_message_dense_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype021
/create_message/dense_9/BiasAdd_1/ReadVariableOpÝ
 create_message/dense_9/BiasAdd_1BiasAdd)create_message/dense_9/MatMul_1:product:07create_message/dense_9/BiasAdd_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	2"
 create_message/dense_9/BiasAdd_1
create_message/dense_9/Tanh_1Tanh)create_message/dense_9/BiasAdd_1:output:0*
T0*
_output_shapes
:	2
create_message/dense_9/Tanh_1¸
PartitionedCall_1PartitionedCall!create_message/dense_9/Tanh_1:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:J * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *-
f(R&
$__inference_message_aggregation_12562
PartitionedCall_1`
concat_3/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat_3/axisĽ
concat_3ConcatV2link_update/dense_12/Tanh:y:0PartitionedCall_1:output:0concat_3/axis:output:0*
N*
T0*
_output_shapes

:J02

concat_3Ń
,link_update/dense_10/MatMul_1/ReadVariableOpReadVariableOp3link_update_dense_10_matmul_readvariableop_resource*
_output_shapes
:	0*
dtype02.
,link_update/dense_10/MatMul_1/ReadVariableOpť
link_update/dense_10/MatMul_1MatMulconcat_3:output:04link_update/dense_10/MatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	J2
link_update/dense_10/MatMul_1Đ
-link_update/dense_10/BiasAdd_1/ReadVariableOpReadVariableOp4link_update_dense_10_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02/
-link_update/dense_10/BiasAdd_1/ReadVariableOpŐ
link_update/dense_10/BiasAdd_1BiasAdd'link_update/dense_10/MatMul_1:product:05link_update/dense_10/BiasAdd_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	J2 
link_update/dense_10/BiasAdd_1
link_update/dense_10/Tanh_1Tanh'link_update/dense_10/BiasAdd_1:output:0*
T0*
_output_shapes
:	J2
link_update/dense_10/Tanh_1Ń
,link_update/dense_11/MatMul_1/ReadVariableOpReadVariableOp3link_update_dense_11_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype02.
,link_update/dense_11/MatMul_1/ReadVariableOpČ
link_update/dense_11/MatMul_1MatMullink_update/dense_10/Tanh_1:y:04link_update/dense_11/MatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes

:J@2
link_update/dense_11/MatMul_1Ď
-link_update/dense_11/BiasAdd_1/ReadVariableOpReadVariableOp4link_update_dense_11_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-link_update/dense_11/BiasAdd_1/ReadVariableOpÔ
link_update/dense_11/BiasAdd_1BiasAdd'link_update/dense_11/MatMul_1:product:05link_update/dense_11/BiasAdd_1/ReadVariableOp:value:0*
T0*
_output_shapes

:J@2 
link_update/dense_11/BiasAdd_1
link_update/dense_11/Tanh_1Tanh'link_update/dense_11/BiasAdd_1:output:0*
T0*
_output_shapes

:J@2
link_update/dense_11/Tanh_1Đ
,link_update/dense_12/MatMul_1/ReadVariableOpReadVariableOp3link_update_dense_12_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02.
,link_update/dense_12/MatMul_1/ReadVariableOpČ
link_update/dense_12/MatMul_1MatMullink_update/dense_11/Tanh_1:y:04link_update/dense_12/MatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes

:J2
link_update/dense_12/MatMul_1Ď
-link_update/dense_12/BiasAdd_1/ReadVariableOpReadVariableOp4link_update_dense_12_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-link_update/dense_12/BiasAdd_1/ReadVariableOpÔ
link_update/dense_12/BiasAdd_1BiasAdd'link_update/dense_12/MatMul_1:product:05link_update/dense_12/BiasAdd_1/ReadVariableOp:value:0*
T0*
_output_shapes

:J2 
link_update/dense_12/BiasAdd_1
link_update/dense_12/Tanh_1Tanh'link_update/dense_12/BiasAdd_1:output:0*
T0*
_output_shapes

:J2
link_update/dense_12/Tanh_1	
GatherV2_4/indicesConst*
_output_shapes	
:*
dtype0*ą
value§B¤"                                                      
   
   
   
   
                                                                                                                                            !   !   !   !   !   !   "   "   "   "   "   "   "   "   #   #   #   %   %   %   %   %   %   &   &   &   '   '   '   '   '   '   '   '   (   (   (   (   )   )   )   )   )   )   )   )   *   *   +   +   +   +   +   +   ,   ,   ,   ,   -   -   -   -   -   -   -   -   .   .   .   .   /   /   /   0   0   0   0   0   1   1   1   1   1   1   1   1   2   2   2   2   3   3   3   3   4   4   4   4   4   5   5   5   6   6   6   6   7   7   7   8   8   8   8   8   8   9   9   9   9   9   9   9   9   :   :   :   :   ;   ;   ;   ;   <   <   =   =   =   =   >   >   >   >   ?   ?   ?   ?   ?   @   @   A   A   A   A   A   A   B   B   B   B   C   C   C   C   C   C   C   C   D   D   D   D   E   E   E   E   F   F   F   F   G   G   G   G   H   H   H   H   H   I   I   I   I   I   I   I   I   2
GatherV2_4/indicesd
GatherV2_4/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2_4/axisÉ

GatherV2_4GatherV2link_update/dense_12/Tanh_1:y:0GatherV2_4/indices:output:0GatherV2_4/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:	2

GatherV2_4	
GatherV2_5/indicesConst*
_output_shapes	
:*
dtype0*ą
value§B¤"      
                                  #   %   &   '   5   6   7   8   9          D   E   F   G   H   I                                                                         !   "                                 B   C   D   E   F   G   H   I   :   ;   <   =   >   ?   @   A         
   D   E   F   G   H   I   2   3   4   :   ;   <   =   >   ?   @   A   *   +   ,   -   :   ;   <   =   >   ?   @   A   (   )   D   E   F   G   H   I   .   /   0   1   :   ;   <   =   >   ?   @   A   *   +   ,   -   2   3   4   5   6   7   8   9   :   ;   <   =   >   ?   @   A   #   %   &   '   .   /   0   1   5   6   7   8   9         
   .   /   0   1   2   3   4   D   E   F   G   H   I   :   ;   <   =   >   ?   @   A          !   "   #   %   &   '   (   )   *   +   ,   -   .   /   0   1   5   6   7   8   9   B   C   D   E   F   G   H   I          !   "   :   ;   <   =   >   ?   @   A                      !   "   #   %   &   '   *   +   ,   -   5   6   7   8   9   :   ;   <   =   >   ?   @   A   2
GatherV2_5/indicesd
GatherV2_5/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2_5/axisÉ

GatherV2_5GatherV2link_update/dense_12/Tanh_1:y:0GatherV2_5/indices:output:0GatherV2_5/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:	2

GatherV2_5`
concat_4/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat_4/axis
concat_4ConcatV2GatherV2_4:output:0GatherV2_5:output:0concat_4/axis:output:0*
N*
T0*
_output_shapes
:	 2

concat_4Ö
.create_message/dense_8/MatMul_2/ReadVariableOpReadVariableOp5create_message_dense_8_matmul_readvariableop_resource*
_output_shapes

: @*
dtype020
.create_message/dense_8/MatMul_2/ReadVariableOpÁ
create_message/dense_8/MatMul_2MatMulconcat_4:output:06create_message/dense_8/MatMul_2/ReadVariableOp:value:0*
T0*
_output_shapes
:	@2!
create_message/dense_8/MatMul_2Ő
/create_message/dense_8/BiasAdd_2/ReadVariableOpReadVariableOp6create_message_dense_8_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype021
/create_message/dense_8/BiasAdd_2/ReadVariableOpÝ
 create_message/dense_8/BiasAdd_2BiasAdd)create_message/dense_8/MatMul_2:product:07create_message/dense_8/BiasAdd_2/ReadVariableOp:value:0*
T0*
_output_shapes
:	@2"
 create_message/dense_8/BiasAdd_2
create_message/dense_8/Tanh_2Tanh)create_message/dense_8/BiasAdd_2:output:0*
T0*
_output_shapes
:	@2
create_message/dense_8/Tanh_2Ö
.create_message/dense_9/MatMul_2/ReadVariableOpReadVariableOp5create_message_dense_9_matmul_readvariableop_resource*
_output_shapes

:@*
dtype020
.create_message/dense_9/MatMul_2/ReadVariableOpŃ
create_message/dense_9/MatMul_2MatMul!create_message/dense_8/Tanh_2:y:06create_message/dense_9/MatMul_2/ReadVariableOp:value:0*
T0*
_output_shapes
:	2!
create_message/dense_9/MatMul_2Ő
/create_message/dense_9/BiasAdd_2/ReadVariableOpReadVariableOp6create_message_dense_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype021
/create_message/dense_9/BiasAdd_2/ReadVariableOpÝ
 create_message/dense_9/BiasAdd_2BiasAdd)create_message/dense_9/MatMul_2:product:07create_message/dense_9/BiasAdd_2/ReadVariableOp:value:0*
T0*
_output_shapes
:	2"
 create_message/dense_9/BiasAdd_2
create_message/dense_9/Tanh_2Tanh)create_message/dense_9/BiasAdd_2:output:0*
T0*
_output_shapes
:	2
create_message/dense_9/Tanh_2¸
PartitionedCall_2PartitionedCall!create_message/dense_9/Tanh_2:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:J * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *-
f(R&
$__inference_message_aggregation_12562
PartitionedCall_2`
concat_5/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat_5/axis§
concat_5ConcatV2link_update/dense_12/Tanh_1:y:0PartitionedCall_2:output:0concat_5/axis:output:0*
N*
T0*
_output_shapes

:J02

concat_5Ń
,link_update/dense_10/MatMul_2/ReadVariableOpReadVariableOp3link_update_dense_10_matmul_readvariableop_resource*
_output_shapes
:	0*
dtype02.
,link_update/dense_10/MatMul_2/ReadVariableOpť
link_update/dense_10/MatMul_2MatMulconcat_5:output:04link_update/dense_10/MatMul_2/ReadVariableOp:value:0*
T0*
_output_shapes
:	J2
link_update/dense_10/MatMul_2Đ
-link_update/dense_10/BiasAdd_2/ReadVariableOpReadVariableOp4link_update_dense_10_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02/
-link_update/dense_10/BiasAdd_2/ReadVariableOpŐ
link_update/dense_10/BiasAdd_2BiasAdd'link_update/dense_10/MatMul_2:product:05link_update/dense_10/BiasAdd_2/ReadVariableOp:value:0*
T0*
_output_shapes
:	J2 
link_update/dense_10/BiasAdd_2
link_update/dense_10/Tanh_2Tanh'link_update/dense_10/BiasAdd_2:output:0*
T0*
_output_shapes
:	J2
link_update/dense_10/Tanh_2Ń
,link_update/dense_11/MatMul_2/ReadVariableOpReadVariableOp3link_update_dense_11_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype02.
,link_update/dense_11/MatMul_2/ReadVariableOpČ
link_update/dense_11/MatMul_2MatMullink_update/dense_10/Tanh_2:y:04link_update/dense_11/MatMul_2/ReadVariableOp:value:0*
T0*
_output_shapes

:J@2
link_update/dense_11/MatMul_2Ď
-link_update/dense_11/BiasAdd_2/ReadVariableOpReadVariableOp4link_update_dense_11_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-link_update/dense_11/BiasAdd_2/ReadVariableOpÔ
link_update/dense_11/BiasAdd_2BiasAdd'link_update/dense_11/MatMul_2:product:05link_update/dense_11/BiasAdd_2/ReadVariableOp:value:0*
T0*
_output_shapes

:J@2 
link_update/dense_11/BiasAdd_2
link_update/dense_11/Tanh_2Tanh'link_update/dense_11/BiasAdd_2:output:0*
T0*
_output_shapes

:J@2
link_update/dense_11/Tanh_2Đ
,link_update/dense_12/MatMul_2/ReadVariableOpReadVariableOp3link_update_dense_12_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02.
,link_update/dense_12/MatMul_2/ReadVariableOpČ
link_update/dense_12/MatMul_2MatMullink_update/dense_11/Tanh_2:y:04link_update/dense_12/MatMul_2/ReadVariableOp:value:0*
T0*
_output_shapes

:J2
link_update/dense_12/MatMul_2Ď
-link_update/dense_12/BiasAdd_2/ReadVariableOpReadVariableOp4link_update_dense_12_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-link_update/dense_12/BiasAdd_2/ReadVariableOpÔ
link_update/dense_12/BiasAdd_2BiasAdd'link_update/dense_12/MatMul_2:product:05link_update/dense_12/BiasAdd_2/ReadVariableOp:value:0*
T0*
_output_shapes

:J2 
link_update/dense_12/BiasAdd_2
link_update/dense_12/Tanh_2Tanh'link_update/dense_12/BiasAdd_2:output:0*
T0*
_output_shapes

:J2
link_update/dense_12/Tanh_2	
GatherV2_6/indicesConst*
_output_shapes	
:*
dtype0*ą
value§B¤"                                                      
   
   
   
   
                                                                                                                                            !   !   !   !   !   !   "   "   "   "   "   "   "   "   #   #   #   %   %   %   %   %   %   &   &   &   '   '   '   '   '   '   '   '   (   (   (   (   )   )   )   )   )   )   )   )   *   *   +   +   +   +   +   +   ,   ,   ,   ,   -   -   -   -   -   -   -   -   .   .   .   .   /   /   /   0   0   0   0   0   1   1   1   1   1   1   1   1   2   2   2   2   3   3   3   3   4   4   4   4   4   5   5   5   6   6   6   6   7   7   7   8   8   8   8   8   8   9   9   9   9   9   9   9   9   :   :   :   :   ;   ;   ;   ;   <   <   =   =   =   =   >   >   >   >   ?   ?   ?   ?   ?   @   @   A   A   A   A   A   A   B   B   B   B   C   C   C   C   C   C   C   C   D   D   D   D   E   E   E   E   F   F   F   F   G   G   G   G   H   H   H   H   H   I   I   I   I   I   I   I   I   2
GatherV2_6/indicesd
GatherV2_6/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2_6/axisÉ

GatherV2_6GatherV2link_update/dense_12/Tanh_2:y:0GatherV2_6/indices:output:0GatherV2_6/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:	2

GatherV2_6	
GatherV2_7/indicesConst*
_output_shapes	
:*
dtype0*ą
value§B¤"      
                                  #   %   &   '   5   6   7   8   9          D   E   F   G   H   I                                                                         !   "                                 B   C   D   E   F   G   H   I   :   ;   <   =   >   ?   @   A         
   D   E   F   G   H   I   2   3   4   :   ;   <   =   >   ?   @   A   *   +   ,   -   :   ;   <   =   >   ?   @   A   (   )   D   E   F   G   H   I   .   /   0   1   :   ;   <   =   >   ?   @   A   *   +   ,   -   2   3   4   5   6   7   8   9   :   ;   <   =   >   ?   @   A   #   %   &   '   .   /   0   1   5   6   7   8   9         
   .   /   0   1   2   3   4   D   E   F   G   H   I   :   ;   <   =   >   ?   @   A          !   "   #   %   &   '   (   )   *   +   ,   -   .   /   0   1   5   6   7   8   9   B   C   D   E   F   G   H   I          !   "   :   ;   <   =   >   ?   @   A                      !   "   #   %   &   '   *   +   ,   -   5   6   7   8   9   :   ;   <   =   >   ?   @   A   2
GatherV2_7/indicesd
GatherV2_7/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2_7/axisÉ

GatherV2_7GatherV2link_update/dense_12/Tanh_2:y:0GatherV2_7/indices:output:0GatherV2_7/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:	2

GatherV2_7`
concat_6/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat_6/axis
concat_6ConcatV2GatherV2_6:output:0GatherV2_7:output:0concat_6/axis:output:0*
N*
T0*
_output_shapes
:	 2

concat_6Ö
.create_message/dense_8/MatMul_3/ReadVariableOpReadVariableOp5create_message_dense_8_matmul_readvariableop_resource*
_output_shapes

: @*
dtype020
.create_message/dense_8/MatMul_3/ReadVariableOpÁ
create_message/dense_8/MatMul_3MatMulconcat_6:output:06create_message/dense_8/MatMul_3/ReadVariableOp:value:0*
T0*
_output_shapes
:	@2!
create_message/dense_8/MatMul_3Ő
/create_message/dense_8/BiasAdd_3/ReadVariableOpReadVariableOp6create_message_dense_8_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype021
/create_message/dense_8/BiasAdd_3/ReadVariableOpÝ
 create_message/dense_8/BiasAdd_3BiasAdd)create_message/dense_8/MatMul_3:product:07create_message/dense_8/BiasAdd_3/ReadVariableOp:value:0*
T0*
_output_shapes
:	@2"
 create_message/dense_8/BiasAdd_3
create_message/dense_8/Tanh_3Tanh)create_message/dense_8/BiasAdd_3:output:0*
T0*
_output_shapes
:	@2
create_message/dense_8/Tanh_3Ö
.create_message/dense_9/MatMul_3/ReadVariableOpReadVariableOp5create_message_dense_9_matmul_readvariableop_resource*
_output_shapes

:@*
dtype020
.create_message/dense_9/MatMul_3/ReadVariableOpŃ
create_message/dense_9/MatMul_3MatMul!create_message/dense_8/Tanh_3:y:06create_message/dense_9/MatMul_3/ReadVariableOp:value:0*
T0*
_output_shapes
:	2!
create_message/dense_9/MatMul_3Ő
/create_message/dense_9/BiasAdd_3/ReadVariableOpReadVariableOp6create_message_dense_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype021
/create_message/dense_9/BiasAdd_3/ReadVariableOpÝ
 create_message/dense_9/BiasAdd_3BiasAdd)create_message/dense_9/MatMul_3:product:07create_message/dense_9/BiasAdd_3/ReadVariableOp:value:0*
T0*
_output_shapes
:	2"
 create_message/dense_9/BiasAdd_3
create_message/dense_9/Tanh_3Tanh)create_message/dense_9/BiasAdd_3:output:0*
T0*
_output_shapes
:	2
create_message/dense_9/Tanh_3¸
PartitionedCall_3PartitionedCall!create_message/dense_9/Tanh_3:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:J * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *-
f(R&
$__inference_message_aggregation_12562
PartitionedCall_3`
concat_7/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat_7/axis§
concat_7ConcatV2link_update/dense_12/Tanh_2:y:0PartitionedCall_3:output:0concat_7/axis:output:0*
N*
T0*
_output_shapes

:J02

concat_7Ń
,link_update/dense_10/MatMul_3/ReadVariableOpReadVariableOp3link_update_dense_10_matmul_readvariableop_resource*
_output_shapes
:	0*
dtype02.
,link_update/dense_10/MatMul_3/ReadVariableOpť
link_update/dense_10/MatMul_3MatMulconcat_7:output:04link_update/dense_10/MatMul_3/ReadVariableOp:value:0*
T0*
_output_shapes
:	J2
link_update/dense_10/MatMul_3Đ
-link_update/dense_10/BiasAdd_3/ReadVariableOpReadVariableOp4link_update_dense_10_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02/
-link_update/dense_10/BiasAdd_3/ReadVariableOpŐ
link_update/dense_10/BiasAdd_3BiasAdd'link_update/dense_10/MatMul_3:product:05link_update/dense_10/BiasAdd_3/ReadVariableOp:value:0*
T0*
_output_shapes
:	J2 
link_update/dense_10/BiasAdd_3
link_update/dense_10/Tanh_3Tanh'link_update/dense_10/BiasAdd_3:output:0*
T0*
_output_shapes
:	J2
link_update/dense_10/Tanh_3Ń
,link_update/dense_11/MatMul_3/ReadVariableOpReadVariableOp3link_update_dense_11_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype02.
,link_update/dense_11/MatMul_3/ReadVariableOpČ
link_update/dense_11/MatMul_3MatMullink_update/dense_10/Tanh_3:y:04link_update/dense_11/MatMul_3/ReadVariableOp:value:0*
T0*
_output_shapes

:J@2
link_update/dense_11/MatMul_3Ď
-link_update/dense_11/BiasAdd_3/ReadVariableOpReadVariableOp4link_update_dense_11_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-link_update/dense_11/BiasAdd_3/ReadVariableOpÔ
link_update/dense_11/BiasAdd_3BiasAdd'link_update/dense_11/MatMul_3:product:05link_update/dense_11/BiasAdd_3/ReadVariableOp:value:0*
T0*
_output_shapes

:J@2 
link_update/dense_11/BiasAdd_3
link_update/dense_11/Tanh_3Tanh'link_update/dense_11/BiasAdd_3:output:0*
T0*
_output_shapes

:J@2
link_update/dense_11/Tanh_3Đ
,link_update/dense_12/MatMul_3/ReadVariableOpReadVariableOp3link_update_dense_12_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02.
,link_update/dense_12/MatMul_3/ReadVariableOpČ
link_update/dense_12/MatMul_3MatMullink_update/dense_11/Tanh_3:y:04link_update/dense_12/MatMul_3/ReadVariableOp:value:0*
T0*
_output_shapes

:J2
link_update/dense_12/MatMul_3Ď
-link_update/dense_12/BiasAdd_3/ReadVariableOpReadVariableOp4link_update_dense_12_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-link_update/dense_12/BiasAdd_3/ReadVariableOpÔ
link_update/dense_12/BiasAdd_3BiasAdd'link_update/dense_12/MatMul_3:product:05link_update/dense_12/BiasAdd_3/ReadVariableOp:value:0*
T0*
_output_shapes

:J2 
link_update/dense_12/BiasAdd_3
link_update/dense_12/Tanh_3Tanh'link_update/dense_12/BiasAdd_3:output:0*
T0*
_output_shapes

:J2
link_update/dense_12/Tanh_3	
GatherV2_8/indicesConst*
_output_shapes	
:*
dtype0*ą
value§B¤"                                                      
   
   
   
   
                                                                                                                                            !   !   !   !   !   !   "   "   "   "   "   "   "   "   #   #   #   %   %   %   %   %   %   &   &   &   '   '   '   '   '   '   '   '   (   (   (   (   )   )   )   )   )   )   )   )   *   *   +   +   +   +   +   +   ,   ,   ,   ,   -   -   -   -   -   -   -   -   .   .   .   .   /   /   /   0   0   0   0   0   1   1   1   1   1   1   1   1   2   2   2   2   3   3   3   3   4   4   4   4   4   5   5   5   6   6   6   6   7   7   7   8   8   8   8   8   8   9   9   9   9   9   9   9   9   :   :   :   :   ;   ;   ;   ;   <   <   =   =   =   =   >   >   >   >   ?   ?   ?   ?   ?   @   @   A   A   A   A   A   A   B   B   B   B   C   C   C   C   C   C   C   C   D   D   D   D   E   E   E   E   F   F   F   F   G   G   G   G   H   H   H   H   H   I   I   I   I   I   I   I   I   2
GatherV2_8/indicesd
GatherV2_8/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2_8/axisÉ

GatherV2_8GatherV2link_update/dense_12/Tanh_3:y:0GatherV2_8/indices:output:0GatherV2_8/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:	2

GatherV2_8	
GatherV2_9/indicesConst*
_output_shapes	
:*
dtype0*ą
value§B¤"      
                                  #   %   &   '   5   6   7   8   9          D   E   F   G   H   I                                                                         !   "                                 B   C   D   E   F   G   H   I   :   ;   <   =   >   ?   @   A         
   D   E   F   G   H   I   2   3   4   :   ;   <   =   >   ?   @   A   *   +   ,   -   :   ;   <   =   >   ?   @   A   (   )   D   E   F   G   H   I   .   /   0   1   :   ;   <   =   >   ?   @   A   *   +   ,   -   2   3   4   5   6   7   8   9   :   ;   <   =   >   ?   @   A   #   %   &   '   .   /   0   1   5   6   7   8   9         
   .   /   0   1   2   3   4   D   E   F   G   H   I   :   ;   <   =   >   ?   @   A          !   "   #   %   &   '   (   )   *   +   ,   -   .   /   0   1   5   6   7   8   9   B   C   D   E   F   G   H   I          !   "   :   ;   <   =   >   ?   @   A                      !   "   #   %   &   '   *   +   ,   -   5   6   7   8   9   :   ;   <   =   >   ?   @   A   2
GatherV2_9/indicesd
GatherV2_9/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2_9/axisÉ

GatherV2_9GatherV2link_update/dense_12/Tanh_3:y:0GatherV2_9/indices:output:0GatherV2_9/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:	2

GatherV2_9`
concat_8/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat_8/axis
concat_8ConcatV2GatherV2_8:output:0GatherV2_9:output:0concat_8/axis:output:0*
N*
T0*
_output_shapes
:	 2

concat_8Ö
.create_message/dense_8/MatMul_4/ReadVariableOpReadVariableOp5create_message_dense_8_matmul_readvariableop_resource*
_output_shapes

: @*
dtype020
.create_message/dense_8/MatMul_4/ReadVariableOpÁ
create_message/dense_8/MatMul_4MatMulconcat_8:output:06create_message/dense_8/MatMul_4/ReadVariableOp:value:0*
T0*
_output_shapes
:	@2!
create_message/dense_8/MatMul_4Ő
/create_message/dense_8/BiasAdd_4/ReadVariableOpReadVariableOp6create_message_dense_8_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype021
/create_message/dense_8/BiasAdd_4/ReadVariableOpÝ
 create_message/dense_8/BiasAdd_4BiasAdd)create_message/dense_8/MatMul_4:product:07create_message/dense_8/BiasAdd_4/ReadVariableOp:value:0*
T0*
_output_shapes
:	@2"
 create_message/dense_8/BiasAdd_4
create_message/dense_8/Tanh_4Tanh)create_message/dense_8/BiasAdd_4:output:0*
T0*
_output_shapes
:	@2
create_message/dense_8/Tanh_4Ö
.create_message/dense_9/MatMul_4/ReadVariableOpReadVariableOp5create_message_dense_9_matmul_readvariableop_resource*
_output_shapes

:@*
dtype020
.create_message/dense_9/MatMul_4/ReadVariableOpŃ
create_message/dense_9/MatMul_4MatMul!create_message/dense_8/Tanh_4:y:06create_message/dense_9/MatMul_4/ReadVariableOp:value:0*
T0*
_output_shapes
:	2!
create_message/dense_9/MatMul_4Ő
/create_message/dense_9/BiasAdd_4/ReadVariableOpReadVariableOp6create_message_dense_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype021
/create_message/dense_9/BiasAdd_4/ReadVariableOpÝ
 create_message/dense_9/BiasAdd_4BiasAdd)create_message/dense_9/MatMul_4:product:07create_message/dense_9/BiasAdd_4/ReadVariableOp:value:0*
T0*
_output_shapes
:	2"
 create_message/dense_9/BiasAdd_4
create_message/dense_9/Tanh_4Tanh)create_message/dense_9/BiasAdd_4:output:0*
T0*
_output_shapes
:	2
create_message/dense_9/Tanh_4¸
PartitionedCall_4PartitionedCall!create_message/dense_9/Tanh_4:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:J * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *-
f(R&
$__inference_message_aggregation_12562
PartitionedCall_4`
concat_9/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat_9/axis§
concat_9ConcatV2link_update/dense_12/Tanh_3:y:0PartitionedCall_4:output:0concat_9/axis:output:0*
N*
T0*
_output_shapes

:J02

concat_9Ń
,link_update/dense_10/MatMul_4/ReadVariableOpReadVariableOp3link_update_dense_10_matmul_readvariableop_resource*
_output_shapes
:	0*
dtype02.
,link_update/dense_10/MatMul_4/ReadVariableOpť
link_update/dense_10/MatMul_4MatMulconcat_9:output:04link_update/dense_10/MatMul_4/ReadVariableOp:value:0*
T0*
_output_shapes
:	J2
link_update/dense_10/MatMul_4Đ
-link_update/dense_10/BiasAdd_4/ReadVariableOpReadVariableOp4link_update_dense_10_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02/
-link_update/dense_10/BiasAdd_4/ReadVariableOpŐ
link_update/dense_10/BiasAdd_4BiasAdd'link_update/dense_10/MatMul_4:product:05link_update/dense_10/BiasAdd_4/ReadVariableOp:value:0*
T0*
_output_shapes
:	J2 
link_update/dense_10/BiasAdd_4
link_update/dense_10/Tanh_4Tanh'link_update/dense_10/BiasAdd_4:output:0*
T0*
_output_shapes
:	J2
link_update/dense_10/Tanh_4Ń
,link_update/dense_11/MatMul_4/ReadVariableOpReadVariableOp3link_update_dense_11_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype02.
,link_update/dense_11/MatMul_4/ReadVariableOpČ
link_update/dense_11/MatMul_4MatMullink_update/dense_10/Tanh_4:y:04link_update/dense_11/MatMul_4/ReadVariableOp:value:0*
T0*
_output_shapes

:J@2
link_update/dense_11/MatMul_4Ď
-link_update/dense_11/BiasAdd_4/ReadVariableOpReadVariableOp4link_update_dense_11_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-link_update/dense_11/BiasAdd_4/ReadVariableOpÔ
link_update/dense_11/BiasAdd_4BiasAdd'link_update/dense_11/MatMul_4:product:05link_update/dense_11/BiasAdd_4/ReadVariableOp:value:0*
T0*
_output_shapes

:J@2 
link_update/dense_11/BiasAdd_4
link_update/dense_11/Tanh_4Tanh'link_update/dense_11/BiasAdd_4:output:0*
T0*
_output_shapes

:J@2
link_update/dense_11/Tanh_4Đ
,link_update/dense_12/MatMul_4/ReadVariableOpReadVariableOp3link_update_dense_12_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02.
,link_update/dense_12/MatMul_4/ReadVariableOpČ
link_update/dense_12/MatMul_4MatMullink_update/dense_11/Tanh_4:y:04link_update/dense_12/MatMul_4/ReadVariableOp:value:0*
T0*
_output_shapes

:J2
link_update/dense_12/MatMul_4Ď
-link_update/dense_12/BiasAdd_4/ReadVariableOpReadVariableOp4link_update_dense_12_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-link_update/dense_12/BiasAdd_4/ReadVariableOpÔ
link_update/dense_12/BiasAdd_4BiasAdd'link_update/dense_12/MatMul_4:product:05link_update/dense_12/BiasAdd_4/ReadVariableOp:value:0*
T0*
_output_shapes

:J2 
link_update/dense_12/BiasAdd_4
link_update/dense_12/Tanh_4Tanh'link_update/dense_12/BiasAdd_4:output:0*
T0*
_output_shapes

:J2
link_update/dense_12/Tanh_4	
GatherV2_10/indicesConst*
_output_shapes	
:*
dtype0*ą
value§B¤"                                                      
   
   
   
   
                                                                                                                                            !   !   !   !   !   !   "   "   "   "   "   "   "   "   #   #   #   %   %   %   %   %   %   &   &   &   '   '   '   '   '   '   '   '   (   (   (   (   )   )   )   )   )   )   )   )   *   *   +   +   +   +   +   +   ,   ,   ,   ,   -   -   -   -   -   -   -   -   .   .   .   .   /   /   /   0   0   0   0   0   1   1   1   1   1   1   1   1   2   2   2   2   3   3   3   3   4   4   4   4   4   5   5   5   6   6   6   6   7   7   7   8   8   8   8   8   8   9   9   9   9   9   9   9   9   :   :   :   :   ;   ;   ;   ;   <   <   =   =   =   =   >   >   >   >   ?   ?   ?   ?   ?   @   @   A   A   A   A   A   A   B   B   B   B   C   C   C   C   C   C   C   C   D   D   D   D   E   E   E   E   F   F   F   F   G   G   G   G   H   H   H   H   H   I   I   I   I   I   I   I   I   2
GatherV2_10/indicesf
GatherV2_10/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2_10/axisÍ
GatherV2_10GatherV2link_update/dense_12/Tanh_4:y:0GatherV2_10/indices:output:0GatherV2_10/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:	2
GatherV2_10	
GatherV2_11/indicesConst*
_output_shapes	
:*
dtype0*ą
value§B¤"      
                                  #   %   &   '   5   6   7   8   9          D   E   F   G   H   I                                                                         !   "                                 B   C   D   E   F   G   H   I   :   ;   <   =   >   ?   @   A         
   D   E   F   G   H   I   2   3   4   :   ;   <   =   >   ?   @   A   *   +   ,   -   :   ;   <   =   >   ?   @   A   (   )   D   E   F   G   H   I   .   /   0   1   :   ;   <   =   >   ?   @   A   *   +   ,   -   2   3   4   5   6   7   8   9   :   ;   <   =   >   ?   @   A   #   %   &   '   .   /   0   1   5   6   7   8   9         
   .   /   0   1   2   3   4   D   E   F   G   H   I   :   ;   <   =   >   ?   @   A          !   "   #   %   &   '   (   )   *   +   ,   -   .   /   0   1   5   6   7   8   9   B   C   D   E   F   G   H   I          !   "   :   ;   <   =   >   ?   @   A                      !   "   #   %   &   '   *   +   ,   -   5   6   7   8   9   :   ;   <   =   >   ?   @   A   2
GatherV2_11/indicesf
GatherV2_11/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2_11/axisÍ
GatherV2_11GatherV2link_update/dense_12/Tanh_4:y:0GatherV2_11/indices:output:0GatherV2_11/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:	2
GatherV2_11b
concat_10/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat_10/axis
	concat_10ConcatV2GatherV2_10:output:0GatherV2_11:output:0concat_10/axis:output:0*
N*
T0*
_output_shapes
:	 2
	concat_10Ö
.create_message/dense_8/MatMul_5/ReadVariableOpReadVariableOp5create_message_dense_8_matmul_readvariableop_resource*
_output_shapes

: @*
dtype020
.create_message/dense_8/MatMul_5/ReadVariableOpÂ
create_message/dense_8/MatMul_5MatMulconcat_10:output:06create_message/dense_8/MatMul_5/ReadVariableOp:value:0*
T0*
_output_shapes
:	@2!
create_message/dense_8/MatMul_5Ő
/create_message/dense_8/BiasAdd_5/ReadVariableOpReadVariableOp6create_message_dense_8_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype021
/create_message/dense_8/BiasAdd_5/ReadVariableOpÝ
 create_message/dense_8/BiasAdd_5BiasAdd)create_message/dense_8/MatMul_5:product:07create_message/dense_8/BiasAdd_5/ReadVariableOp:value:0*
T0*
_output_shapes
:	@2"
 create_message/dense_8/BiasAdd_5
create_message/dense_8/Tanh_5Tanh)create_message/dense_8/BiasAdd_5:output:0*
T0*
_output_shapes
:	@2
create_message/dense_8/Tanh_5Ö
.create_message/dense_9/MatMul_5/ReadVariableOpReadVariableOp5create_message_dense_9_matmul_readvariableop_resource*
_output_shapes

:@*
dtype020
.create_message/dense_9/MatMul_5/ReadVariableOpŃ
create_message/dense_9/MatMul_5MatMul!create_message/dense_8/Tanh_5:y:06create_message/dense_9/MatMul_5/ReadVariableOp:value:0*
T0*
_output_shapes
:	2!
create_message/dense_9/MatMul_5Ő
/create_message/dense_9/BiasAdd_5/ReadVariableOpReadVariableOp6create_message_dense_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype021
/create_message/dense_9/BiasAdd_5/ReadVariableOpÝ
 create_message/dense_9/BiasAdd_5BiasAdd)create_message/dense_9/MatMul_5:product:07create_message/dense_9/BiasAdd_5/ReadVariableOp:value:0*
T0*
_output_shapes
:	2"
 create_message/dense_9/BiasAdd_5
create_message/dense_9/Tanh_5Tanh)create_message/dense_9/BiasAdd_5:output:0*
T0*
_output_shapes
:	2
create_message/dense_9/Tanh_5¸
PartitionedCall_5PartitionedCall!create_message/dense_9/Tanh_5:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:J * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *-
f(R&
$__inference_message_aggregation_12562
PartitionedCall_5b
concat_11/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat_11/axisŞ
	concat_11ConcatV2link_update/dense_12/Tanh_4:y:0PartitionedCall_5:output:0concat_11/axis:output:0*
N*
T0*
_output_shapes

:J02
	concat_11Ń
,link_update/dense_10/MatMul_5/ReadVariableOpReadVariableOp3link_update_dense_10_matmul_readvariableop_resource*
_output_shapes
:	0*
dtype02.
,link_update/dense_10/MatMul_5/ReadVariableOpź
link_update/dense_10/MatMul_5MatMulconcat_11:output:04link_update/dense_10/MatMul_5/ReadVariableOp:value:0*
T0*
_output_shapes
:	J2
link_update/dense_10/MatMul_5Đ
-link_update/dense_10/BiasAdd_5/ReadVariableOpReadVariableOp4link_update_dense_10_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02/
-link_update/dense_10/BiasAdd_5/ReadVariableOpŐ
link_update/dense_10/BiasAdd_5BiasAdd'link_update/dense_10/MatMul_5:product:05link_update/dense_10/BiasAdd_5/ReadVariableOp:value:0*
T0*
_output_shapes
:	J2 
link_update/dense_10/BiasAdd_5
link_update/dense_10/Tanh_5Tanh'link_update/dense_10/BiasAdd_5:output:0*
T0*
_output_shapes
:	J2
link_update/dense_10/Tanh_5Ń
,link_update/dense_11/MatMul_5/ReadVariableOpReadVariableOp3link_update_dense_11_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype02.
,link_update/dense_11/MatMul_5/ReadVariableOpČ
link_update/dense_11/MatMul_5MatMullink_update/dense_10/Tanh_5:y:04link_update/dense_11/MatMul_5/ReadVariableOp:value:0*
T0*
_output_shapes

:J@2
link_update/dense_11/MatMul_5Ď
-link_update/dense_11/BiasAdd_5/ReadVariableOpReadVariableOp4link_update_dense_11_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-link_update/dense_11/BiasAdd_5/ReadVariableOpÔ
link_update/dense_11/BiasAdd_5BiasAdd'link_update/dense_11/MatMul_5:product:05link_update/dense_11/BiasAdd_5/ReadVariableOp:value:0*
T0*
_output_shapes

:J@2 
link_update/dense_11/BiasAdd_5
link_update/dense_11/Tanh_5Tanh'link_update/dense_11/BiasAdd_5:output:0*
T0*
_output_shapes

:J@2
link_update/dense_11/Tanh_5Đ
,link_update/dense_12/MatMul_5/ReadVariableOpReadVariableOp3link_update_dense_12_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02.
,link_update/dense_12/MatMul_5/ReadVariableOpČ
link_update/dense_12/MatMul_5MatMullink_update/dense_11/Tanh_5:y:04link_update/dense_12/MatMul_5/ReadVariableOp:value:0*
T0*
_output_shapes

:J2
link_update/dense_12/MatMul_5Ď
-link_update/dense_12/BiasAdd_5/ReadVariableOpReadVariableOp4link_update_dense_12_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-link_update/dense_12/BiasAdd_5/ReadVariableOpÔ
link_update/dense_12/BiasAdd_5BiasAdd'link_update/dense_12/MatMul_5:product:05link_update/dense_12/BiasAdd_5/ReadVariableOp:value:0*
T0*
_output_shapes

:J2 
link_update/dense_12/BiasAdd_5
link_update/dense_12/Tanh_5Tanh'link_update/dense_12/BiasAdd_5:output:0*
T0*
_output_shapes

:J2
link_update/dense_12/Tanh_5	
GatherV2_12/indicesConst*
_output_shapes	
:*
dtype0*ą
value§B¤"                                                      
   
   
   
   
                                                                                                                                            !   !   !   !   !   !   "   "   "   "   "   "   "   "   #   #   #   %   %   %   %   %   %   &   &   &   '   '   '   '   '   '   '   '   (   (   (   (   )   )   )   )   )   )   )   )   *   *   +   +   +   +   +   +   ,   ,   ,   ,   -   -   -   -   -   -   -   -   .   .   .   .   /   /   /   0   0   0   0   0   1   1   1   1   1   1   1   1   2   2   2   2   3   3   3   3   4   4   4   4   4   5   5   5   6   6   6   6   7   7   7   8   8   8   8   8   8   9   9   9   9   9   9   9   9   :   :   :   :   ;   ;   ;   ;   <   <   =   =   =   =   >   >   >   >   ?   ?   ?   ?   ?   @   @   A   A   A   A   A   A   B   B   B   B   C   C   C   C   C   C   C   C   D   D   D   D   E   E   E   E   F   F   F   F   G   G   G   G   H   H   H   H   H   I   I   I   I   I   I   I   I   2
GatherV2_12/indicesf
GatherV2_12/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2_12/axisÍ
GatherV2_12GatherV2link_update/dense_12/Tanh_5:y:0GatherV2_12/indices:output:0GatherV2_12/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:	2
GatherV2_12	
GatherV2_13/indicesConst*
_output_shapes	
:*
dtype0*ą
value§B¤"      
                                  #   %   &   '   5   6   7   8   9          D   E   F   G   H   I                                                                         !   "                                 B   C   D   E   F   G   H   I   :   ;   <   =   >   ?   @   A         
   D   E   F   G   H   I   2   3   4   :   ;   <   =   >   ?   @   A   *   +   ,   -   :   ;   <   =   >   ?   @   A   (   )   D   E   F   G   H   I   .   /   0   1   :   ;   <   =   >   ?   @   A   *   +   ,   -   2   3   4   5   6   7   8   9   :   ;   <   =   >   ?   @   A   #   %   &   '   .   /   0   1   5   6   7   8   9         
   .   /   0   1   2   3   4   D   E   F   G   H   I   :   ;   <   =   >   ?   @   A          !   "   #   %   &   '   (   )   *   +   ,   -   .   /   0   1   5   6   7   8   9   B   C   D   E   F   G   H   I          !   "   :   ;   <   =   >   ?   @   A                      !   "   #   %   &   '   *   +   ,   -   5   6   7   8   9   :   ;   <   =   >   ?   @   A   2
GatherV2_13/indicesf
GatherV2_13/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2_13/axisÍ
GatherV2_13GatherV2link_update/dense_12/Tanh_5:y:0GatherV2_13/indices:output:0GatherV2_13/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:	2
GatherV2_13b
concat_12/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat_12/axis
	concat_12ConcatV2GatherV2_12:output:0GatherV2_13:output:0concat_12/axis:output:0*
N*
T0*
_output_shapes
:	 2
	concat_12Ö
.create_message/dense_8/MatMul_6/ReadVariableOpReadVariableOp5create_message_dense_8_matmul_readvariableop_resource*
_output_shapes

: @*
dtype020
.create_message/dense_8/MatMul_6/ReadVariableOpÂ
create_message/dense_8/MatMul_6MatMulconcat_12:output:06create_message/dense_8/MatMul_6/ReadVariableOp:value:0*
T0*
_output_shapes
:	@2!
create_message/dense_8/MatMul_6Ő
/create_message/dense_8/BiasAdd_6/ReadVariableOpReadVariableOp6create_message_dense_8_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype021
/create_message/dense_8/BiasAdd_6/ReadVariableOpÝ
 create_message/dense_8/BiasAdd_6BiasAdd)create_message/dense_8/MatMul_6:product:07create_message/dense_8/BiasAdd_6/ReadVariableOp:value:0*
T0*
_output_shapes
:	@2"
 create_message/dense_8/BiasAdd_6
create_message/dense_8/Tanh_6Tanh)create_message/dense_8/BiasAdd_6:output:0*
T0*
_output_shapes
:	@2
create_message/dense_8/Tanh_6Ö
.create_message/dense_9/MatMul_6/ReadVariableOpReadVariableOp5create_message_dense_9_matmul_readvariableop_resource*
_output_shapes

:@*
dtype020
.create_message/dense_9/MatMul_6/ReadVariableOpŃ
create_message/dense_9/MatMul_6MatMul!create_message/dense_8/Tanh_6:y:06create_message/dense_9/MatMul_6/ReadVariableOp:value:0*
T0*
_output_shapes
:	2!
create_message/dense_9/MatMul_6Ő
/create_message/dense_9/BiasAdd_6/ReadVariableOpReadVariableOp6create_message_dense_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype021
/create_message/dense_9/BiasAdd_6/ReadVariableOpÝ
 create_message/dense_9/BiasAdd_6BiasAdd)create_message/dense_9/MatMul_6:product:07create_message/dense_9/BiasAdd_6/ReadVariableOp:value:0*
T0*
_output_shapes
:	2"
 create_message/dense_9/BiasAdd_6
create_message/dense_9/Tanh_6Tanh)create_message/dense_9/BiasAdd_6:output:0*
T0*
_output_shapes
:	2
create_message/dense_9/Tanh_6¸
PartitionedCall_6PartitionedCall!create_message/dense_9/Tanh_6:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:J * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *-
f(R&
$__inference_message_aggregation_12562
PartitionedCall_6b
concat_13/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat_13/axisŞ
	concat_13ConcatV2link_update/dense_12/Tanh_5:y:0PartitionedCall_6:output:0concat_13/axis:output:0*
N*
T0*
_output_shapes

:J02
	concat_13Ń
,link_update/dense_10/MatMul_6/ReadVariableOpReadVariableOp3link_update_dense_10_matmul_readvariableop_resource*
_output_shapes
:	0*
dtype02.
,link_update/dense_10/MatMul_6/ReadVariableOpź
link_update/dense_10/MatMul_6MatMulconcat_13:output:04link_update/dense_10/MatMul_6/ReadVariableOp:value:0*
T0*
_output_shapes
:	J2
link_update/dense_10/MatMul_6Đ
-link_update/dense_10/BiasAdd_6/ReadVariableOpReadVariableOp4link_update_dense_10_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02/
-link_update/dense_10/BiasAdd_6/ReadVariableOpŐ
link_update/dense_10/BiasAdd_6BiasAdd'link_update/dense_10/MatMul_6:product:05link_update/dense_10/BiasAdd_6/ReadVariableOp:value:0*
T0*
_output_shapes
:	J2 
link_update/dense_10/BiasAdd_6
link_update/dense_10/Tanh_6Tanh'link_update/dense_10/BiasAdd_6:output:0*
T0*
_output_shapes
:	J2
link_update/dense_10/Tanh_6Ń
,link_update/dense_11/MatMul_6/ReadVariableOpReadVariableOp3link_update_dense_11_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype02.
,link_update/dense_11/MatMul_6/ReadVariableOpČ
link_update/dense_11/MatMul_6MatMullink_update/dense_10/Tanh_6:y:04link_update/dense_11/MatMul_6/ReadVariableOp:value:0*
T0*
_output_shapes

:J@2
link_update/dense_11/MatMul_6Ď
-link_update/dense_11/BiasAdd_6/ReadVariableOpReadVariableOp4link_update_dense_11_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-link_update/dense_11/BiasAdd_6/ReadVariableOpÔ
link_update/dense_11/BiasAdd_6BiasAdd'link_update/dense_11/MatMul_6:product:05link_update/dense_11/BiasAdd_6/ReadVariableOp:value:0*
T0*
_output_shapes

:J@2 
link_update/dense_11/BiasAdd_6
link_update/dense_11/Tanh_6Tanh'link_update/dense_11/BiasAdd_6:output:0*
T0*
_output_shapes

:J@2
link_update/dense_11/Tanh_6Đ
,link_update/dense_12/MatMul_6/ReadVariableOpReadVariableOp3link_update_dense_12_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02.
,link_update/dense_12/MatMul_6/ReadVariableOpČ
link_update/dense_12/MatMul_6MatMullink_update/dense_11/Tanh_6:y:04link_update/dense_12/MatMul_6/ReadVariableOp:value:0*
T0*
_output_shapes

:J2
link_update/dense_12/MatMul_6Ď
-link_update/dense_12/BiasAdd_6/ReadVariableOpReadVariableOp4link_update_dense_12_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-link_update/dense_12/BiasAdd_6/ReadVariableOpÔ
link_update/dense_12/BiasAdd_6BiasAdd'link_update/dense_12/MatMul_6:product:05link_update/dense_12/BiasAdd_6/ReadVariableOp:value:0*
T0*
_output_shapes

:J2 
link_update/dense_12/BiasAdd_6
link_update/dense_12/Tanh_6Tanh'link_update/dense_12/BiasAdd_6:output:0*
T0*
_output_shapes

:J2
link_update/dense_12/Tanh_6	
GatherV2_14/indicesConst*
_output_shapes	
:*
dtype0*ą
value§B¤"                                                      
   
   
   
   
                                                                                                                                            !   !   !   !   !   !   "   "   "   "   "   "   "   "   #   #   #   %   %   %   %   %   %   &   &   &   '   '   '   '   '   '   '   '   (   (   (   (   )   )   )   )   )   )   )   )   *   *   +   +   +   +   +   +   ,   ,   ,   ,   -   -   -   -   -   -   -   -   .   .   .   .   /   /   /   0   0   0   0   0   1   1   1   1   1   1   1   1   2   2   2   2   3   3   3   3   4   4   4   4   4   5   5   5   6   6   6   6   7   7   7   8   8   8   8   8   8   9   9   9   9   9   9   9   9   :   :   :   :   ;   ;   ;   ;   <   <   =   =   =   =   >   >   >   >   ?   ?   ?   ?   ?   @   @   A   A   A   A   A   A   B   B   B   B   C   C   C   C   C   C   C   C   D   D   D   D   E   E   E   E   F   F   F   F   G   G   G   G   H   H   H   H   H   I   I   I   I   I   I   I   I   2
GatherV2_14/indicesf
GatherV2_14/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2_14/axisÍ
GatherV2_14GatherV2link_update/dense_12/Tanh_6:y:0GatherV2_14/indices:output:0GatherV2_14/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:	2
GatherV2_14	
GatherV2_15/indicesConst*
_output_shapes	
:*
dtype0*ą
value§B¤"      
                                  #   %   &   '   5   6   7   8   9          D   E   F   G   H   I                                                                         !   "                                 B   C   D   E   F   G   H   I   :   ;   <   =   >   ?   @   A         
   D   E   F   G   H   I   2   3   4   :   ;   <   =   >   ?   @   A   *   +   ,   -   :   ;   <   =   >   ?   @   A   (   )   D   E   F   G   H   I   .   /   0   1   :   ;   <   =   >   ?   @   A   *   +   ,   -   2   3   4   5   6   7   8   9   :   ;   <   =   >   ?   @   A   #   %   &   '   .   /   0   1   5   6   7   8   9         
   .   /   0   1   2   3   4   D   E   F   G   H   I   :   ;   <   =   >   ?   @   A          !   "   #   %   &   '   (   )   *   +   ,   -   .   /   0   1   5   6   7   8   9   B   C   D   E   F   G   H   I          !   "   :   ;   <   =   >   ?   @   A                      !   "   #   %   &   '   *   +   ,   -   5   6   7   8   9   :   ;   <   =   >   ?   @   A   2
GatherV2_15/indicesf
GatherV2_15/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2_15/axisÍ
GatherV2_15GatherV2link_update/dense_12/Tanh_6:y:0GatherV2_15/indices:output:0GatherV2_15/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:	2
GatherV2_15b
concat_14/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat_14/axis
	concat_14ConcatV2GatherV2_14:output:0GatherV2_15:output:0concat_14/axis:output:0*
N*
T0*
_output_shapes
:	 2
	concat_14Ö
.create_message/dense_8/MatMul_7/ReadVariableOpReadVariableOp5create_message_dense_8_matmul_readvariableop_resource*
_output_shapes

: @*
dtype020
.create_message/dense_8/MatMul_7/ReadVariableOpÂ
create_message/dense_8/MatMul_7MatMulconcat_14:output:06create_message/dense_8/MatMul_7/ReadVariableOp:value:0*
T0*
_output_shapes
:	@2!
create_message/dense_8/MatMul_7Ő
/create_message/dense_8/BiasAdd_7/ReadVariableOpReadVariableOp6create_message_dense_8_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype021
/create_message/dense_8/BiasAdd_7/ReadVariableOpÝ
 create_message/dense_8/BiasAdd_7BiasAdd)create_message/dense_8/MatMul_7:product:07create_message/dense_8/BiasAdd_7/ReadVariableOp:value:0*
T0*
_output_shapes
:	@2"
 create_message/dense_8/BiasAdd_7
create_message/dense_8/Tanh_7Tanh)create_message/dense_8/BiasAdd_7:output:0*
T0*
_output_shapes
:	@2
create_message/dense_8/Tanh_7Ö
.create_message/dense_9/MatMul_7/ReadVariableOpReadVariableOp5create_message_dense_9_matmul_readvariableop_resource*
_output_shapes

:@*
dtype020
.create_message/dense_9/MatMul_7/ReadVariableOpŃ
create_message/dense_9/MatMul_7MatMul!create_message/dense_8/Tanh_7:y:06create_message/dense_9/MatMul_7/ReadVariableOp:value:0*
T0*
_output_shapes
:	2!
create_message/dense_9/MatMul_7Ő
/create_message/dense_9/BiasAdd_7/ReadVariableOpReadVariableOp6create_message_dense_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype021
/create_message/dense_9/BiasAdd_7/ReadVariableOpÝ
 create_message/dense_9/BiasAdd_7BiasAdd)create_message/dense_9/MatMul_7:product:07create_message/dense_9/BiasAdd_7/ReadVariableOp:value:0*
T0*
_output_shapes
:	2"
 create_message/dense_9/BiasAdd_7
create_message/dense_9/Tanh_7Tanh)create_message/dense_9/BiasAdd_7:output:0*
T0*
_output_shapes
:	2
create_message/dense_9/Tanh_7¸
PartitionedCall_7PartitionedCall!create_message/dense_9/Tanh_7:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:J * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *-
f(R&
$__inference_message_aggregation_12562
PartitionedCall_7b
concat_15/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat_15/axisŞ
	concat_15ConcatV2link_update/dense_12/Tanh_6:y:0PartitionedCall_7:output:0concat_15/axis:output:0*
N*
T0*
_output_shapes

:J02
	concat_15Ń
,link_update/dense_10/MatMul_7/ReadVariableOpReadVariableOp3link_update_dense_10_matmul_readvariableop_resource*
_output_shapes
:	0*
dtype02.
,link_update/dense_10/MatMul_7/ReadVariableOpź
link_update/dense_10/MatMul_7MatMulconcat_15:output:04link_update/dense_10/MatMul_7/ReadVariableOp:value:0*
T0*
_output_shapes
:	J2
link_update/dense_10/MatMul_7Đ
-link_update/dense_10/BiasAdd_7/ReadVariableOpReadVariableOp4link_update_dense_10_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02/
-link_update/dense_10/BiasAdd_7/ReadVariableOpŐ
link_update/dense_10/BiasAdd_7BiasAdd'link_update/dense_10/MatMul_7:product:05link_update/dense_10/BiasAdd_7/ReadVariableOp:value:0*
T0*
_output_shapes
:	J2 
link_update/dense_10/BiasAdd_7
link_update/dense_10/Tanh_7Tanh'link_update/dense_10/BiasAdd_7:output:0*
T0*
_output_shapes
:	J2
link_update/dense_10/Tanh_7Ń
,link_update/dense_11/MatMul_7/ReadVariableOpReadVariableOp3link_update_dense_11_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype02.
,link_update/dense_11/MatMul_7/ReadVariableOpČ
link_update/dense_11/MatMul_7MatMullink_update/dense_10/Tanh_7:y:04link_update/dense_11/MatMul_7/ReadVariableOp:value:0*
T0*
_output_shapes

:J@2
link_update/dense_11/MatMul_7Ď
-link_update/dense_11/BiasAdd_7/ReadVariableOpReadVariableOp4link_update_dense_11_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-link_update/dense_11/BiasAdd_7/ReadVariableOpÔ
link_update/dense_11/BiasAdd_7BiasAdd'link_update/dense_11/MatMul_7:product:05link_update/dense_11/BiasAdd_7/ReadVariableOp:value:0*
T0*
_output_shapes

:J@2 
link_update/dense_11/BiasAdd_7
link_update/dense_11/Tanh_7Tanh'link_update/dense_11/BiasAdd_7:output:0*
T0*
_output_shapes

:J@2
link_update/dense_11/Tanh_7Đ
,link_update/dense_12/MatMul_7/ReadVariableOpReadVariableOp3link_update_dense_12_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02.
,link_update/dense_12/MatMul_7/ReadVariableOpČ
link_update/dense_12/MatMul_7MatMullink_update/dense_11/Tanh_7:y:04link_update/dense_12/MatMul_7/ReadVariableOp:value:0*
T0*
_output_shapes

:J2
link_update/dense_12/MatMul_7Ď
-link_update/dense_12/BiasAdd_7/ReadVariableOpReadVariableOp4link_update_dense_12_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-link_update/dense_12/BiasAdd_7/ReadVariableOpÔ
link_update/dense_12/BiasAdd_7BiasAdd'link_update/dense_12/MatMul_7:product:05link_update/dense_12/BiasAdd_7/ReadVariableOp:value:0*
T0*
_output_shapes

:J2 
link_update/dense_12/BiasAdd_7
link_update/dense_12/Tanh_7Tanh'link_update/dense_12/BiasAdd_7:output:0*
T0*
_output_shapes

:J2
link_update/dense_12/Tanh_7j
IdentityIdentitylink_update/dense_12/Tanh_7:y:0*
T0*
_output_shapes

:J2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:˙˙˙˙˙˙˙˙˙:::::::::::J F
#
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_nameinput

Ť
C__inference_dense_12_layer_call_and_return_conditional_losses_55224

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2	
BiasAddX
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2
Tanh\
IdentityIdentityTanh:y:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*.
_input_shapes
:˙˙˙˙˙˙˙˙˙@:::O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙@
 
_user_specified_nameinputs
%
Ł
__inference_call_43875	
input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_83
/readout_dense_13_matmul_readvariableop_resource4
0readout_dense_13_biasadd_readvariableop_resource3
/readout_dense_14_matmul_readvariableop_resource4
0readout_dense_14_biasadd_readvariableop_resource3
/readout_dense_15_matmul_readvariableop_resource4
0readout_dense_15_biasadd_readvariableop_resource
identity˘StatefulPartitionedCallŽ
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:J*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8 *)
f$R"
 __inference_message_passing_15342
StatefulPartitionedCallś
PartitionedCallPartitionedCall StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *0
f+R)
'__inference_generate_readout_input_15772
PartitionedCallÁ
&readout/dense_13/MatMul/ReadVariableOpReadVariableOp/readout_dense_13_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype02(
&readout/dense_13/MatMul/ReadVariableOp°
readout/dense_13/MatMulMatMulPartitionedCall:output:0.readout/dense_13/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
readout/dense_13/MatMulŔ
'readout/dense_13/BiasAdd/ReadVariableOpReadVariableOp0readout_dense_13_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02)
'readout/dense_13/BiasAdd/ReadVariableOp˝
readout/dense_13/BiasAddBiasAdd!readout/dense_13/MatMul:product:0/readout/dense_13/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
readout/dense_13/BiasAdd
readout/dense_13/TanhTanh!readout/dense_13/BiasAdd:output:0*
T0*
_output_shapes
:	2
readout/dense_13/Tanh
readout/dropout_2/IdentityIdentityreadout/dense_13/Tanh:y:0*
T0*
_output_shapes
:	2
readout/dropout_2/IdentityÁ
&readout/dense_14/MatMul/ReadVariableOpReadVariableOp/readout_dense_14_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype02(
&readout/dense_14/MatMul/ReadVariableOpş
readout/dense_14/MatMulMatMul#readout/dropout_2/Identity:output:0.readout/dense_14/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:@2
readout/dense_14/MatMulż
'readout/dense_14/BiasAdd/ReadVariableOpReadVariableOp0readout_dense_14_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02)
'readout/dense_14/BiasAdd/ReadVariableOpź
readout/dense_14/BiasAddBiasAdd!readout/dense_14/MatMul:product:0/readout/dense_14/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:@2
readout/dense_14/BiasAdd
readout/dense_14/TanhTanh!readout/dense_14/BiasAdd:output:0*
T0*
_output_shapes

:@2
readout/dense_14/Tanh
readout/dropout_3/IdentityIdentityreadout/dense_14/Tanh:y:0*
T0*
_output_shapes

:@2
readout/dropout_3/IdentityŔ
&readout/dense_15/MatMul/ReadVariableOpReadVariableOp/readout_dense_15_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02(
&readout/dense_15/MatMul/ReadVariableOpş
readout/dense_15/MatMulMatMul#readout/dropout_3/Identity:output:0.readout/dense_15/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
readout/dense_15/MatMulż
'readout/dense_15/BiasAdd/ReadVariableOpReadVariableOp0readout_dense_15_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'readout/dense_15/BiasAdd/ReadVariableOpź
readout/dense_15/BiasAddBiasAdd!readout/dense_15/MatMul:product:0/readout/dense_15/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:2
readout/dense_15/BiasAddq
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙2
Reshape/shape}
ReshapeReshape!readout/dense_15/BiasAdd:output:0Reshape/shape:output:0*
T0*
_output_shapes
:2	
Reshapeq
IdentityIdentityReshape:output:0^StatefulPartitionedCall*
T0*
_output_shapes
:2

Identity"
identityIdentity:output:0*Z
_input_shapesI
G:::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:B >

_output_shapes	
:

_user_specified_nameinput
¤
Ť
C__inference_dense_10_layer_call_and_return_conditional_losses_53667

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	0*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2	
BiasAddY
TanhTanhBiasAdd:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
Tanh]
IdentityIdentityTanh:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*.
_input_shapes
:˙˙˙˙˙˙˙˙˙0:::O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙0
 
_user_specified_nameinputs
Ă
˙
I__inference_create_message_layer_call_and_return_conditional_losses_53597
dense_8_input
dense_8_53586
dense_8_53588
dense_9_53591
dense_9_53593
identity˘dense_8/StatefulPartitionedCall˘dense_9/StatefulPartitionedCall
dense_8/StatefulPartitionedCallStatefulPartitionedCalldense_8_inputdense_8_53586dense_8_53588*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_8_layer_call_and_return_conditional_losses_535392!
dense_8/StatefulPartitionedCallŽ
dense_9/StatefulPartitionedCallStatefulPartitionedCall(dense_8/StatefulPartitionedCall:output:0dense_9_53591dense_9_53593*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_9_layer_call_and_return_conditional_losses_535662!
dense_9/StatefulPartitionedCallŔ
IdentityIdentity(dense_9/StatefulPartitionedCall:output:0 ^dense_8/StatefulPartitionedCall ^dense_9/StatefulPartitionedCall*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:˙˙˙˙˙˙˙˙˙ ::::2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall:V R
'
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
'
_user_specified_namedense_8_input


Ó
&__inference_critic_layer_call_fn_54562
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14
identity˘StatefulPartitionedCall 
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_critic_layer_call_and_return_conditional_losses_543002
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes
:2

Identity"
identityIdentity:output:0*b
_input_shapesQ
O:˙˙˙˙˙˙˙˙˙::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:L H
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
!
_user_specified_name	input_1
Ž
ř
I__inference_create_message_layer_call_and_return_conditional_losses_53641

inputs
dense_8_53630
dense_8_53632
dense_9_53635
dense_9_53637
identity˘dense_8/StatefulPartitionedCall˘dense_9/StatefulPartitionedCall
dense_8/StatefulPartitionedCallStatefulPartitionedCallinputsdense_8_53630dense_8_53632*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_8_layer_call_and_return_conditional_losses_535392!
dense_8/StatefulPartitionedCallŽ
dense_9/StatefulPartitionedCallStatefulPartitionedCall(dense_8/StatefulPartitionedCall:output:0dense_9_53635dense_9_53637*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_9_layer_call_and_return_conditional_losses_535662!
dense_9/StatefulPartitionedCallŔ
IdentityIdentity(dense_9/StatefulPartitionedCall:output:0 ^dense_8/StatefulPartitionedCall ^dense_9/StatefulPartitionedCall*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:˙˙˙˙˙˙˙˙˙ ::::2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall:O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
 
_user_specified_nameinputs
+
ň
B__inference_readout_layer_call_and_return_conditional_losses_55073

inputs+
'dense_13_matmul_readvariableop_resource,
(dense_13_biasadd_readvariableop_resource+
'dense_14_matmul_readvariableop_resource,
(dense_14_biasadd_readvariableop_resource+
'dense_15_matmul_readvariableop_resource,
(dense_15_biasadd_readvariableop_resource
identityŠ
dense_13/MatMul/ReadVariableOpReadVariableOp'dense_13_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype02 
dense_13/MatMul/ReadVariableOp
dense_13/MatMulMatMulinputs&dense_13/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
dense_13/MatMul¨
dense_13/BiasAdd/ReadVariableOpReadVariableOp(dense_13_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02!
dense_13/BiasAdd/ReadVariableOpŚ
dense_13/BiasAddBiasAdddense_13/MatMul:product:0'dense_13/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
dense_13/BiasAddt
dense_13/TanhTanhdense_13/BiasAdd:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
dense_13/Tanhw
dropout_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?2
dropout_2/dropout/Const
dropout_2/dropout/MulMuldense_13/Tanh:y:0 dropout_2/dropout/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
dropout_2/dropout/Muls
dropout_2/dropout/ShapeShapedense_13/Tanh:y:0*
T0*
_output_shapes
:2
dropout_2/dropout/ShapeÓ
.dropout_2/dropout/random_uniform/RandomUniformRandomUniform dropout_2/dropout/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype020
.dropout_2/dropout/random_uniform/RandomUniform
 dropout_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2"
 dropout_2/dropout/GreaterEqual/yç
dropout_2/dropout/GreaterEqualGreaterEqual7dropout_2/dropout/random_uniform/RandomUniform:output:0)dropout_2/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2 
dropout_2/dropout/GreaterEqual
dropout_2/dropout/CastCast"dropout_2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
dropout_2/dropout/CastŁ
dropout_2/dropout/Mul_1Muldropout_2/dropout/Mul:z:0dropout_2/dropout/Cast:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
dropout_2/dropout/Mul_1Š
dense_14/MatMul/ReadVariableOpReadVariableOp'dense_14_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype02 
dense_14/MatMul/ReadVariableOpŁ
dense_14/MatMulMatMuldropout_2/dropout/Mul_1:z:0&dense_14/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@2
dense_14/MatMul§
dense_14/BiasAdd/ReadVariableOpReadVariableOp(dense_14_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
dense_14/BiasAdd/ReadVariableOpĽ
dense_14/BiasAddBiasAdddense_14/MatMul:product:0'dense_14/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@2
dense_14/BiasAdds
dense_14/TanhTanhdense_14/BiasAdd:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@2
dense_14/Tanhw
dropout_3/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?2
dropout_3/dropout/Const
dropout_3/dropout/MulMuldense_14/Tanh:y:0 dropout_3/dropout/Const:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@2
dropout_3/dropout/Muls
dropout_3/dropout/ShapeShapedense_14/Tanh:y:0*
T0*
_output_shapes
:2
dropout_3/dropout/ShapeŇ
.dropout_3/dropout/random_uniform/RandomUniformRandomUniform dropout_3/dropout/Shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@*
dtype020
.dropout_3/dropout/random_uniform/RandomUniform
 dropout_3/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2"
 dropout_3/dropout/GreaterEqual/yć
dropout_3/dropout/GreaterEqualGreaterEqual7dropout_3/dropout/random_uniform/RandomUniform:output:0)dropout_3/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@2 
dropout_3/dropout/GreaterEqual
dropout_3/dropout/CastCast"dropout_3/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@2
dropout_3/dropout/Cast˘
dropout_3/dropout/Mul_1Muldropout_3/dropout/Mul:z:0dropout_3/dropout/Cast:y:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@2
dropout_3/dropout/Mul_1¨
dense_15/MatMul/ReadVariableOpReadVariableOp'dense_15_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02 
dense_15/MatMul/ReadVariableOpŁ
dense_15/MatMulMatMuldropout_3/dropout/Mul_1:z:0&dense_15/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2
dense_15/MatMul§
dense_15/BiasAdd/ReadVariableOpReadVariableOp(dense_15_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_15/BiasAdd/ReadVariableOpĽ
dense_15/BiasAddBiasAdddense_15/MatMul:product:0'dense_15/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2
dense_15/BiasAddm
IdentityIdentitydense_15/BiasAdd:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:˙˙˙˙˙˙˙˙˙@:::::::O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙@
 
_user_specified_nameinputs
ľ
ö
F__inference_link_update_layer_call_and_return_conditional_losses_54899

inputs+
'dense_10_matmul_readvariableop_resource,
(dense_10_biasadd_readvariableop_resource+
'dense_11_matmul_readvariableop_resource,
(dense_11_biasadd_readvariableop_resource+
'dense_12_matmul_readvariableop_resource,
(dense_12_biasadd_readvariableop_resource
identityŠ
dense_10/MatMul/ReadVariableOpReadVariableOp'dense_10_matmul_readvariableop_resource*
_output_shapes
:	0*
dtype02 
dense_10/MatMul/ReadVariableOp
dense_10/MatMulMatMulinputs&dense_10/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
dense_10/MatMul¨
dense_10/BiasAdd/ReadVariableOpReadVariableOp(dense_10_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02!
dense_10/BiasAdd/ReadVariableOpŚ
dense_10/BiasAddBiasAdddense_10/MatMul:product:0'dense_10/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
dense_10/BiasAddt
dense_10/TanhTanhdense_10/BiasAdd:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
dense_10/TanhŠ
dense_11/MatMul/ReadVariableOpReadVariableOp'dense_11_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype02 
dense_11/MatMul/ReadVariableOp
dense_11/MatMulMatMuldense_10/Tanh:y:0&dense_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@2
dense_11/MatMul§
dense_11/BiasAdd/ReadVariableOpReadVariableOp(dense_11_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
dense_11/BiasAdd/ReadVariableOpĽ
dense_11/BiasAddBiasAdddense_11/MatMul:product:0'dense_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@2
dense_11/BiasAdds
dense_11/TanhTanhdense_11/BiasAdd:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@2
dense_11/Tanh¨
dense_12/MatMul/ReadVariableOpReadVariableOp'dense_12_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02 
dense_12/MatMul/ReadVariableOp
dense_12/MatMulMatMuldense_11/Tanh:y:0&dense_12/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2
dense_12/MatMul§
dense_12/BiasAdd/ReadVariableOpReadVariableOp(dense_12_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_12/BiasAdd/ReadVariableOpĽ
dense_12/BiasAddBiasAdddense_12/MatMul:product:0'dense_12/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2
dense_12/BiasAdds
dense_12/TanhTanhdense_12/BiasAdd:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2
dense_12/Tanhe
IdentityIdentitydense_12/Tanh:y:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:˙˙˙˙˙˙˙˙˙0:::::::O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙0
 
_user_specified_nameinputs
§
Ą
.__inference_create_message_layer_call_fn_54836

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity˘StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_create_message_layer_call_and_return_conditional_losses_536142
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:˙˙˙˙˙˙˙˙˙ ::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
 
_user_specified_nameinputs
Ă%
Î
A__inference_critic_layer_call_and_return_conditional_losses_54713	
input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_83
/readout_dense_13_matmul_readvariableop_resource4
0readout_dense_13_biasadd_readvariableop_resource3
/readout_dense_14_matmul_readvariableop_resource4
0readout_dense_14_biasadd_readvariableop_resource3
/readout_dense_15_matmul_readvariableop_resource4
0readout_dense_15_biasadd_readvariableop_resource
identity˘StatefulPartitionedCallŻ
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:J*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8 **
f%R#
!__inference_message_passing_424792
StatefulPartitionedCallś
PartitionedCallPartitionedCall StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *0
f+R)
'__inference_generate_readout_input_15772
PartitionedCallÁ
&readout/dense_13/MatMul/ReadVariableOpReadVariableOp/readout_dense_13_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype02(
&readout/dense_13/MatMul/ReadVariableOp°
readout/dense_13/MatMulMatMulPartitionedCall:output:0.readout/dense_13/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
readout/dense_13/MatMulŔ
'readout/dense_13/BiasAdd/ReadVariableOpReadVariableOp0readout_dense_13_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02)
'readout/dense_13/BiasAdd/ReadVariableOp˝
readout/dense_13/BiasAddBiasAdd!readout/dense_13/MatMul:product:0/readout/dense_13/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
readout/dense_13/BiasAdd
readout/dense_13/TanhTanh!readout/dense_13/BiasAdd:output:0*
T0*
_output_shapes
:	2
readout/dense_13/Tanh
readout/dropout_2/IdentityIdentityreadout/dense_13/Tanh:y:0*
T0*
_output_shapes
:	2
readout/dropout_2/IdentityÁ
&readout/dense_14/MatMul/ReadVariableOpReadVariableOp/readout_dense_14_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype02(
&readout/dense_14/MatMul/ReadVariableOpş
readout/dense_14/MatMulMatMul#readout/dropout_2/Identity:output:0.readout/dense_14/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:@2
readout/dense_14/MatMulż
'readout/dense_14/BiasAdd/ReadVariableOpReadVariableOp0readout_dense_14_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02)
'readout/dense_14/BiasAdd/ReadVariableOpź
readout/dense_14/BiasAddBiasAdd!readout/dense_14/MatMul:product:0/readout/dense_14/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:@2
readout/dense_14/BiasAdd
readout/dense_14/TanhTanh!readout/dense_14/BiasAdd:output:0*
T0*
_output_shapes

:@2
readout/dense_14/Tanh
readout/dropout_3/IdentityIdentityreadout/dense_14/Tanh:y:0*
T0*
_output_shapes

:@2
readout/dropout_3/IdentityŔ
&readout/dense_15/MatMul/ReadVariableOpReadVariableOp/readout_dense_15_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02(
&readout/dense_15/MatMul/ReadVariableOpş
readout/dense_15/MatMulMatMul#readout/dropout_3/Identity:output:0.readout/dense_15/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
readout/dense_15/MatMulż
'readout/dense_15/BiasAdd/ReadVariableOpReadVariableOp0readout_dense_15_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'readout/dense_15/BiasAdd/ReadVariableOpź
readout/dense_15/BiasAddBiasAdd!readout/dense_15/MatMul:product:0/readout/dense_15/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:2
readout/dense_15/BiasAddq
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙2
Reshape/shape}
ReshapeReshape!readout/dense_15/BiasAdd:output:0Reshape/shape:output:0*
T0*
_output_shapes
:2	
Reshapeq
IdentityIdentityReshape:output:0^StatefulPartitionedCall*
T0*
_output_shapes
:2

Identity"
identityIdentity:output:0*b
_input_shapesQ
O:˙˙˙˙˙˙˙˙˙::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:J F
#
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_nameinput
Ü
}
(__inference_dense_11_layer_call_fn_55213

inputs
unknown
	unknown_0
identity˘StatefulPartitionedCalló
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_11_layer_call_and_return_conditional_losses_536942
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@2

Identity"
identityIdentity:output:0*/
_input_shapes
:˙˙˙˙˙˙˙˙˙::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Ř
|
'__inference_dense_8_layer_call_fn_55153

inputs
unknown
	unknown_0
identity˘StatefulPartitionedCallň
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_8_layer_call_and_return_conditional_losses_535392
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@2

Identity"
identityIdentity:output:0*.
_input_shapes
:˙˙˙˙˙˙˙˙˙ ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
 
_user_specified_nameinputs
ş
C
%__inference_message_aggregation_43958
messages
identity§	
UnsortedSegmentMax/segment_idsConst*
_output_shapes	
:*
dtype0*ą
value§B¤"      
                                  #   %   &   '   5   6   7   8   9          D   E   F   G   H   I                                                                         !   "                                 B   C   D   E   F   G   H   I   :   ;   <   =   >   ?   @   A         
   D   E   F   G   H   I   2   3   4   :   ;   <   =   >   ?   @   A   *   +   ,   -   :   ;   <   =   >   ?   @   A   (   )   D   E   F   G   H   I   .   /   0   1   :   ;   <   =   >   ?   @   A   *   +   ,   -   2   3   4   5   6   7   8   9   :   ;   <   =   >   ?   @   A   #   %   &   '   .   /   0   1   5   6   7   8   9         
   .   /   0   1   2   3   4   D   E   F   G   H   I   :   ;   <   =   >   ?   @   A          !   "   #   %   &   '   (   )   *   +   ,   -   .   /   0   1   5   6   7   8   9   B   C   D   E   F   G   H   I          !   "   :   ;   <   =   >   ?   @   A                      !   "   #   %   &   '   *   +   ,   -   5   6   7   8   9   :   ;   <   =   >   ?   @   A   2 
UnsortedSegmentMax/segment_ids
UnsortedSegmentMax/num_segmentsConst*
_output_shapes
: *
dtype0*
value	B :J2!
UnsortedSegmentMax/num_segmentsÔ
UnsortedSegmentMaxUnsortedSegmentMaxmessages'UnsortedSegmentMax/segment_ids:output:0(UnsortedSegmentMax/num_segments:output:0*
T0*
Tindices0*
_output_shapes

:J2
UnsortedSegmentMax§	
UnsortedSegmentMin/segment_idsConst*
_output_shapes	
:*
dtype0*ą
value§B¤"      
                                  #   %   &   '   5   6   7   8   9          D   E   F   G   H   I                                                                         !   "                                 B   C   D   E   F   G   H   I   :   ;   <   =   >   ?   @   A         
   D   E   F   G   H   I   2   3   4   :   ;   <   =   >   ?   @   A   *   +   ,   -   :   ;   <   =   >   ?   @   A   (   )   D   E   F   G   H   I   .   /   0   1   :   ;   <   =   >   ?   @   A   *   +   ,   -   2   3   4   5   6   7   8   9   :   ;   <   =   >   ?   @   A   #   %   &   '   .   /   0   1   5   6   7   8   9         
   .   /   0   1   2   3   4   D   E   F   G   H   I   :   ;   <   =   >   ?   @   A          !   "   #   %   &   '   (   )   *   +   ,   -   .   /   0   1   5   6   7   8   9   B   C   D   E   F   G   H   I          !   "   :   ;   <   =   >   ?   @   A                      !   "   #   %   &   '   *   +   ,   -   5   6   7   8   9   :   ;   <   =   >   ?   @   A   2 
UnsortedSegmentMin/segment_ids
UnsortedSegmentMin/num_segmentsConst*
_output_shapes
: *
dtype0*
value	B :J2!
UnsortedSegmentMin/num_segmentsÔ
UnsortedSegmentMinUnsortedSegmentMinmessages'UnsortedSegmentMin/segment_ids:output:0(UnsortedSegmentMin/num_segments:output:0*
T0*
Tindices0*
_output_shapes

:J2
UnsortedSegmentMin\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis
concatConcatV2UnsortedSegmentMax:output:0UnsortedSegmentMin:output:0concat/axis:output:0*
N*
T0*
_output_shapes

:J 2
concatZ
IdentityIdentityconcat:output:0*
T0*
_output_shapes

:J 2

Identity"
identityIdentity:output:0*
_input_shapes
:	:I E

_output_shapes
:	
"
_user_specified_name
messages
˘
b
)__inference_dropout_2_layer_call_fn_55275

inputs
identity˘StatefulPartitionedCallŰ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_2_layer_call_and_return_conditional_losses_538732
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*'
_input_shapes
:˙˙˙˙˙˙˙˙˙22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
ľ
ö
F__inference_link_update_layer_call_and_return_conditional_losses_54874

inputs+
'dense_10_matmul_readvariableop_resource,
(dense_10_biasadd_readvariableop_resource+
'dense_11_matmul_readvariableop_resource,
(dense_11_biasadd_readvariableop_resource+
'dense_12_matmul_readvariableop_resource,
(dense_12_biasadd_readvariableop_resource
identityŠ
dense_10/MatMul/ReadVariableOpReadVariableOp'dense_10_matmul_readvariableop_resource*
_output_shapes
:	0*
dtype02 
dense_10/MatMul/ReadVariableOp
dense_10/MatMulMatMulinputs&dense_10/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
dense_10/MatMul¨
dense_10/BiasAdd/ReadVariableOpReadVariableOp(dense_10_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02!
dense_10/BiasAdd/ReadVariableOpŚ
dense_10/BiasAddBiasAdddense_10/MatMul:product:0'dense_10/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
dense_10/BiasAddt
dense_10/TanhTanhdense_10/BiasAdd:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
dense_10/TanhŠ
dense_11/MatMul/ReadVariableOpReadVariableOp'dense_11_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype02 
dense_11/MatMul/ReadVariableOp
dense_11/MatMulMatMuldense_10/Tanh:y:0&dense_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@2
dense_11/MatMul§
dense_11/BiasAdd/ReadVariableOpReadVariableOp(dense_11_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
dense_11/BiasAdd/ReadVariableOpĽ
dense_11/BiasAddBiasAdddense_11/MatMul:product:0'dense_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@2
dense_11/BiasAdds
dense_11/TanhTanhdense_11/BiasAdd:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@2
dense_11/Tanh¨
dense_12/MatMul/ReadVariableOpReadVariableOp'dense_12_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02 
dense_12/MatMul/ReadVariableOp
dense_12/MatMulMatMuldense_11/Tanh:y:0&dense_12/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2
dense_12/MatMul§
dense_12/BiasAdd/ReadVariableOpReadVariableOp(dense_12_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_12/BiasAdd/ReadVariableOpĽ
dense_12/BiasAddBiasAdddense_12/MatMul:product:0'dense_12/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2
dense_12/BiasAdds
dense_12/TanhTanhdense_12/BiasAdd:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2
dense_12/Tanhe
IdentityIdentitydense_12/Tanh:y:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:˙˙˙˙˙˙˙˙˙0:::::::O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙0
 
_user_specified_nameinputs
Ă
˙
I__inference_create_message_layer_call_and_return_conditional_losses_53583
dense_8_input
dense_8_53550
dense_8_53552
dense_9_53577
dense_9_53579
identity˘dense_8/StatefulPartitionedCall˘dense_9/StatefulPartitionedCall
dense_8/StatefulPartitionedCallStatefulPartitionedCalldense_8_inputdense_8_53550dense_8_53552*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_8_layer_call_and_return_conditional_losses_535392!
dense_8/StatefulPartitionedCallŽ
dense_9/StatefulPartitionedCallStatefulPartitionedCall(dense_8/StatefulPartitionedCall:output:0dense_9_53577dense_9_53579*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_9_layer_call_and_return_conditional_losses_535662!
dense_9/StatefulPartitionedCallŔ
IdentityIdentity(dense_9/StatefulPartitionedCall:output:0 ^dense_8/StatefulPartitionedCall ^dense_9/StatefulPartitionedCall*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:˙˙˙˙˙˙˙˙˙ ::::2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall:V R
'
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
'
_user_specified_namedense_8_input
ş

B__inference_readout_layer_call_and_return_conditional_losses_53975
dense_13_input
dense_13_53856
dense_13_53858
dense_14_53913
dense_14_53915
dense_15_53969
dense_15_53971
identity˘ dense_13/StatefulPartitionedCall˘ dense_14/StatefulPartitionedCall˘ dense_15/StatefulPartitionedCall˘!dropout_2/StatefulPartitionedCall˘!dropout_3/StatefulPartitionedCall
 dense_13/StatefulPartitionedCallStatefulPartitionedCalldense_13_inputdense_13_53856dense_13_53858*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_13_layer_call_and_return_conditional_losses_538452"
 dense_13/StatefulPartitionedCall
!dropout_2/StatefulPartitionedCallStatefulPartitionedCall)dense_13/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_2_layer_call_and_return_conditional_losses_538732#
!dropout_2/StatefulPartitionedCallľ
 dense_14/StatefulPartitionedCallStatefulPartitionedCall*dropout_2/StatefulPartitionedCall:output:0dense_14_53913dense_14_53915*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_14_layer_call_and_return_conditional_losses_539022"
 dense_14/StatefulPartitionedCallľ
!dropout_3/StatefulPartitionedCallStatefulPartitionedCall)dense_14/StatefulPartitionedCall:output:0"^dropout_2/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_3_layer_call_and_return_conditional_losses_539302#
!dropout_3/StatefulPartitionedCallľ
 dense_15/StatefulPartitionedCallStatefulPartitionedCall*dropout_3/StatefulPartitionedCall:output:0dense_15_53969dense_15_53971*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_15_layer_call_and_return_conditional_losses_539582"
 dense_15/StatefulPartitionedCallŽ
IdentityIdentity)dense_15/StatefulPartitionedCall:output:0!^dense_13/StatefulPartitionedCall!^dense_14/StatefulPartitionedCall!^dense_15/StatefulPartitionedCall"^dropout_2/StatefulPartitionedCall"^dropout_3/StatefulPartitionedCall*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:˙˙˙˙˙˙˙˙˙@::::::2D
 dense_13/StatefulPartitionedCall dense_13/StatefulPartitionedCall2D
 dense_14/StatefulPartitionedCall dense_14/StatefulPartitionedCall2D
 dense_15/StatefulPartitionedCall dense_15/StatefulPartitionedCall2F
!dropout_2/StatefulPartitionedCall!dropout_2/StatefulPartitionedCall2F
!dropout_3/StatefulPartitionedCall!dropout_3/StatefulPartitionedCall:W S
'
_output_shapes
:˙˙˙˙˙˙˙˙˙@
(
_user_specified_namedense_13_input
í
I
(__inference_generate_readout_input_43946
link_states
identityr
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 2
Mean/reduction_indicesg
MeanMeanlink_statesMean/reduction_indices:output:0*
T0*
_output_shapes
:2
Meanp
Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 2
Max/reduction_indicesc
MaxMaxlink_statesMax/reduction_indices:output:0*
T0*
_output_shapes
:2
Maxp
Min/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 2
Min/reduction_indicesc
MinMinlink_statesMin/reduction_indices:output:0*
T0*
_output_shapes
:2
Min¨
1reduce_std/reduce_variance/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 23
1reduce_std/reduce_variance/Mean/reduction_indicesÍ
reduce_std/reduce_variance/MeanMeanlink_states:reduce_std/reduce_variance/Mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2!
reduce_std/reduce_variance/Mean§
reduce_std/reduce_variance/subSublink_states(reduce_std/reduce_variance/Mean:output:0*
T0*
_output_shapes

:J2 
reduce_std/reduce_variance/sub
!reduce_std/reduce_variance/SquareSquare"reduce_std/reduce_variance/sub:z:0*
T0*
_output_shapes

:J2#
!reduce_std/reduce_variance/SquareŹ
3reduce_std/reduce_variance/Mean_1/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 25
3reduce_std/reduce_variance/Mean_1/reduction_indicesŘ
!reduce_std/reduce_variance/Mean_1Mean%reduce_std/reduce_variance/Square:y:0<reduce_std/reduce_variance/Mean_1/reduction_indices:output:0*
T0*
_output_shapes
:2#
!reduce_std/reduce_variance/Mean_1{
reduce_std/SqrtSqrt*reduce_std/reduce_variance/Mean_1:output:0*
T0*
_output_shapes
:2
reduce_std/Sqrt\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat/axis 
concatConcatV2Mean:output:0Max:output:0Min:output:0reduce_std/Sqrt:y:0concat/axis:output:0*
N*
T0*
_output_shapes
:@2
concatb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims/dimy

ExpandDims
ExpandDimsconcat:output:0ExpandDims/dim:output:0*
T0*
_output_shapes

:@2

ExpandDims^
IdentityIdentityExpandDims:output:0*
T0*
_output_shapes

:@2

Identity"
identityIdentity:output:0*
_input_shapes

:J:K G

_output_shapes

:J
%
_user_specified_namelink_states
Ů
¸
'__inference_readout_layer_call_fn_55133

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity˘StatefulPartitionedCallŚ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_readout_layer_call_and_return_conditional_losses_540582
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:˙˙˙˙˙˙˙˙˙@::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙@
 
_user_specified_nameinputs
Ü
}
(__inference_dense_14_layer_call_fn_55300

inputs
unknown
	unknown_0
identity˘StatefulPartitionedCalló
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_14_layer_call_and_return_conditional_losses_539022
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@2

Identity"
identityIdentity:output:0*/
_input_shapes
:˙˙˙˙˙˙˙˙˙::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Ą
Ť
C__inference_dense_14_layer_call_and_return_conditional_losses_55291

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@2	
BiasAddX
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@2
Tanh\
IdentityIdentityTanh:y:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@2

Identity"
identityIdentity:output:0*/
_input_shapes
:˙˙˙˙˙˙˙˙˙:::P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Ú
ň
B__inference_readout_layer_call_and_return_conditional_losses_54164

inputs+
'dense_13_matmul_readvariableop_resource,
(dense_13_biasadd_readvariableop_resource+
'dense_14_matmul_readvariableop_resource,
(dense_14_biasadd_readvariableop_resource+
'dense_15_matmul_readvariableop_resource,
(dense_15_biasadd_readvariableop_resource
identityŠ
dense_13/MatMul/ReadVariableOpReadVariableOp'dense_13_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype02 
dense_13/MatMul/ReadVariableOp
dense_13/MatMulMatMulinputs&dense_13/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
dense_13/MatMul¨
dense_13/BiasAdd/ReadVariableOpReadVariableOp(dense_13_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02!
dense_13/BiasAdd/ReadVariableOp
dense_13/BiasAddBiasAdddense_13/MatMul:product:0'dense_13/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
dense_13/BiasAddk
dense_13/TanhTanhdense_13/BiasAdd:output:0*
T0*
_output_shapes
:	2
dense_13/Tanhq
dropout_2/IdentityIdentitydense_13/Tanh:y:0*
T0*
_output_shapes
:	2
dropout_2/IdentityŠ
dense_14/MatMul/ReadVariableOpReadVariableOp'dense_14_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype02 
dense_14/MatMul/ReadVariableOp
dense_14/MatMulMatMuldropout_2/Identity:output:0&dense_14/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:@2
dense_14/MatMul§
dense_14/BiasAdd/ReadVariableOpReadVariableOp(dense_14_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
dense_14/BiasAdd/ReadVariableOp
dense_14/BiasAddBiasAdddense_14/MatMul:product:0'dense_14/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:@2
dense_14/BiasAddj
dense_14/TanhTanhdense_14/BiasAdd:output:0*
T0*
_output_shapes

:@2
dense_14/Tanhp
dropout_3/IdentityIdentitydense_14/Tanh:y:0*
T0*
_output_shapes

:@2
dropout_3/Identity¨
dense_15/MatMul/ReadVariableOpReadVariableOp'dense_15_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02 
dense_15/MatMul/ReadVariableOp
dense_15/MatMulMatMuldropout_3/Identity:output:0&dense_15/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
dense_15/MatMul§
dense_15/BiasAdd/ReadVariableOpReadVariableOp(dense_15_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_15/BiasAdd/ReadVariableOp
dense_15/BiasAddBiasAdddense_15/MatMul:product:0'dense_15/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:2
dense_15/BiasAddd
IdentityIdentitydense_15/BiasAdd:output:0*
T0*
_output_shapes

:2

Identity"
identityIdentity:output:0*5
_input_shapes$
":@:::::::F B

_output_shapes

:@
 
_user_specified_nameinputs
%
Ł
__inference_call_42527	
input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_83
/readout_dense_13_matmul_readvariableop_resource4
0readout_dense_13_biasadd_readvariableop_resource3
/readout_dense_14_matmul_readvariableop_resource4
0readout_dense_14_biasadd_readvariableop_resource3
/readout_dense_15_matmul_readvariableop_resource4
0readout_dense_15_biasadd_readvariableop_resource
identity˘StatefulPartitionedCallŻ
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:J*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8 **
f%R#
!__inference_message_passing_424792
StatefulPartitionedCallś
PartitionedCallPartitionedCall StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *0
f+R)
'__inference_generate_readout_input_15772
PartitionedCallÁ
&readout/dense_13/MatMul/ReadVariableOpReadVariableOp/readout_dense_13_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype02(
&readout/dense_13/MatMul/ReadVariableOp°
readout/dense_13/MatMulMatMulPartitionedCall:output:0.readout/dense_13/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
readout/dense_13/MatMulŔ
'readout/dense_13/BiasAdd/ReadVariableOpReadVariableOp0readout_dense_13_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02)
'readout/dense_13/BiasAdd/ReadVariableOp˝
readout/dense_13/BiasAddBiasAdd!readout/dense_13/MatMul:product:0/readout/dense_13/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
readout/dense_13/BiasAdd
readout/dense_13/TanhTanh!readout/dense_13/BiasAdd:output:0*
T0*
_output_shapes
:	2
readout/dense_13/Tanh
readout/dropout_2/IdentityIdentityreadout/dense_13/Tanh:y:0*
T0*
_output_shapes
:	2
readout/dropout_2/IdentityÁ
&readout/dense_14/MatMul/ReadVariableOpReadVariableOp/readout_dense_14_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype02(
&readout/dense_14/MatMul/ReadVariableOpş
readout/dense_14/MatMulMatMul#readout/dropout_2/Identity:output:0.readout/dense_14/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:@2
readout/dense_14/MatMulż
'readout/dense_14/BiasAdd/ReadVariableOpReadVariableOp0readout_dense_14_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02)
'readout/dense_14/BiasAdd/ReadVariableOpź
readout/dense_14/BiasAddBiasAdd!readout/dense_14/MatMul:product:0/readout/dense_14/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:@2
readout/dense_14/BiasAdd
readout/dense_14/TanhTanh!readout/dense_14/BiasAdd:output:0*
T0*
_output_shapes

:@2
readout/dense_14/Tanh
readout/dropout_3/IdentityIdentityreadout/dense_14/Tanh:y:0*
T0*
_output_shapes

:@2
readout/dropout_3/IdentityŔ
&readout/dense_15/MatMul/ReadVariableOpReadVariableOp/readout_dense_15_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02(
&readout/dense_15/MatMul/ReadVariableOpş
readout/dense_15/MatMulMatMul#readout/dropout_3/Identity:output:0.readout/dense_15/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
readout/dense_15/MatMulż
'readout/dense_15/BiasAdd/ReadVariableOpReadVariableOp0readout_dense_15_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'readout/dense_15/BiasAdd/ReadVariableOpź
readout/dense_15/BiasAddBiasAdd!readout/dense_15/MatMul:product:0/readout/dense_15/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:2
readout/dense_15/BiasAddq
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙2
Reshape/shape}
ReshapeReshape!readout/dense_15/BiasAdd:output:0Reshape/shape:output:0*
T0*
_output_shapes
:2	
Reshapeq
IdentityIdentityReshape:output:0^StatefulPartitionedCall*
T0*
_output_shapes
:2

Identity"
identityIdentity:output:0*b
_input_shapesQ
O:˙˙˙˙˙˙˙˙˙::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:J F
#
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_nameinput
Ú
}
(__inference_dense_15_layer_call_fn_55346

inputs
unknown
	unknown_0
identity˘StatefulPartitionedCalló
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_15_layer_call_and_return_conditional_losses_539582
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*.
_input_shapes
:˙˙˙˙˙˙˙˙˙@::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙@
 
_user_specified_nameinputs
Ě
Ť
C__inference_dense_15_layer_call_and_return_conditional_losses_53958

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2	
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*.
_input_shapes
:˙˙˙˙˙˙˙˙˙@:::O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙@
 
_user_specified_nameinputs
Ë
b
D__inference_dropout_2_layer_call_and_return_conditional_losses_55270

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity_1"!

identity_1Identity_1:output:0*'
_input_shapes
:˙˙˙˙˙˙˙˙˙:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs


I__inference_create_message_layer_call_and_return_conditional_losses_54823

inputs*
&dense_8_matmul_readvariableop_resource+
'dense_8_biasadd_readvariableop_resource*
&dense_9_matmul_readvariableop_resource+
'dense_9_biasadd_readvariableop_resource
identityĽ
dense_8/MatMul/ReadVariableOpReadVariableOp&dense_8_matmul_readvariableop_resource*
_output_shapes

: @*
dtype02
dense_8/MatMul/ReadVariableOp
dense_8/MatMulMatMulinputs%dense_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@2
dense_8/MatMul¤
dense_8/BiasAdd/ReadVariableOpReadVariableOp'dense_8_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02 
dense_8/BiasAdd/ReadVariableOpĄ
dense_8/BiasAddBiasAdddense_8/MatMul:product:0&dense_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@2
dense_8/BiasAddp
dense_8/TanhTanhdense_8/BiasAdd:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@2
dense_8/TanhĽ
dense_9/MatMul/ReadVariableOpReadVariableOp&dense_9_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02
dense_9/MatMul/ReadVariableOp
dense_9/MatMulMatMuldense_8/Tanh:y:0%dense_9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2
dense_9/MatMul¤
dense_9/BiasAdd/ReadVariableOpReadVariableOp'dense_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_9/BiasAdd/ReadVariableOpĄ
dense_9/BiasAddBiasAdddense_9/MatMul:product:0&dense_9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2
dense_9/BiasAddp
dense_9/TanhTanhdense_9/BiasAdd:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2
dense_9/Tanhd
IdentityIdentitydense_9/Tanh:y:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:˙˙˙˙˙˙˙˙˙ :::::O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
 
_user_specified_nameinputs

c
D__inference_dropout_2_layer_call_and_return_conditional_losses_55265

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shapeľ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2
dropout/GreaterEqual/yż
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*'
_input_shapes
:˙˙˙˙˙˙˙˙˙:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
¨D

!__inference__traced_restore_55475
file_prefix#
assignvariableop_dense_8_kernel#
assignvariableop_1_dense_8_bias%
!assignvariableop_2_dense_9_kernel#
assignvariableop_3_dense_9_bias&
"assignvariableop_4_dense_10_kernel$
 assignvariableop_5_dense_10_bias&
"assignvariableop_6_dense_11_kernel$
 assignvariableop_7_dense_11_bias&
"assignvariableop_8_dense_12_kernel$
 assignvariableop_9_dense_12_bias'
#assignvariableop_10_dense_13_kernel%
!assignvariableop_11_dense_13_bias'
#assignvariableop_12_dense_14_kernel%
!assignvariableop_13_dense_14_bias'
#assignvariableop_14_dense_15_kernel%
!assignvariableop_15_dense_15_bias
identity_17˘AssignVariableOp˘AssignVariableOp_1˘AssignVariableOp_10˘AssignVariableOp_11˘AssignVariableOp_12˘AssignVariableOp_13˘AssignVariableOp_14˘AssignVariableOp_15˘AssignVariableOp_2˘AssignVariableOp_3˘AssignVariableOp_4˘AssignVariableOp_5˘AssignVariableOp_6˘AssignVariableOp_7˘AssignVariableOp_8˘AssignVariableOp_9Í
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Ů
valueĎBĚB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/13/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/14/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/15/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names°
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*5
value,B*B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*X
_output_shapesF
D:::::::::::::::::*
dtypes
22
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity
AssignVariableOpAssignVariableOpassignvariableop_dense_8_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1¤
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_8_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2Ś
AssignVariableOp_2AssignVariableOp!assignvariableop_2_dense_9_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3¤
AssignVariableOp_3AssignVariableOpassignvariableop_3_dense_9_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4§
AssignVariableOp_4AssignVariableOp"assignvariableop_4_dense_10_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5Ľ
AssignVariableOp_5AssignVariableOp assignvariableop_5_dense_10_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6§
AssignVariableOp_6AssignVariableOp"assignvariableop_6_dense_11_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7Ľ
AssignVariableOp_7AssignVariableOp assignvariableop_7_dense_11_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8§
AssignVariableOp_8AssignVariableOp"assignvariableop_8_dense_12_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9Ľ
AssignVariableOp_9AssignVariableOp assignvariableop_9_dense_12_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10Ť
AssignVariableOp_10AssignVariableOp#assignvariableop_10_dense_13_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11Š
AssignVariableOp_11AssignVariableOp!assignvariableop_11_dense_13_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12Ť
AssignVariableOp_12AssignVariableOp#assignvariableop_12_dense_14_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13Š
AssignVariableOp_13AssignVariableOp!assignvariableop_13_dense_14_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14Ť
AssignVariableOp_14AssignVariableOp#assignvariableop_14_dense_15_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15Š
AssignVariableOp_15AssignVariableOp!assignvariableop_15_dense_15_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_159
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpž
Identity_16Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_16ą
Identity_17IdentityIdentity_16:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_17"#
identity_17Identity_17:output:0*U
_input_shapesD
B: ::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
á
ź
+__inference_link_update_layer_call_fn_54916

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity˘StatefulPartitionedCallŞ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_link_update_layer_call_and_return_conditional_losses_537792
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:˙˙˙˙˙˙˙˙˙0::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙0
 
_user_specified_nameinputs
Ž
ř
I__inference_create_message_layer_call_and_return_conditional_losses_53614

inputs
dense_8_53603
dense_8_53605
dense_9_53608
dense_9_53610
identity˘dense_8/StatefulPartitionedCall˘dense_9/StatefulPartitionedCall
dense_8/StatefulPartitionedCallStatefulPartitionedCallinputsdense_8_53603dense_8_53605*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_8_layer_call_and_return_conditional_losses_535392!
dense_8/StatefulPartitionedCallŽ
dense_9/StatefulPartitionedCallStatefulPartitionedCall(dense_8/StatefulPartitionedCall:output:0dense_9_53608dense_9_53610*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_9_layer_call_and_return_conditional_losses_535662!
dense_9/StatefulPartitionedCallŔ
IdentityIdentity(dense_9/StatefulPartitionedCall:output:0 ^dense_8/StatefulPartitionedCall ^dense_9/StatefulPartitionedCall*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:˙˙˙˙˙˙˙˙˙ ::::2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall:O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
 
_user_specified_nameinputs

Ş
B__inference_dense_8_layer_call_and_return_conditional_losses_55144

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: @*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@2	
BiasAddX
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@2
Tanh\
IdentityIdentityTanh:y:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@2

Identity"
identityIdentity:output:0*.
_input_shapes
:˙˙˙˙˙˙˙˙˙ :::O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
 
_user_specified_nameinputs
ź°

!__inference_message_passing_42479	
input9
5create_message_dense_8_matmul_readvariableop_resource:
6create_message_dense_8_biasadd_readvariableop_resource9
5create_message_dense_9_matmul_readvariableop_resource:
6create_message_dense_9_biasadd_readvariableop_resource7
3link_update_dense_10_matmul_readvariableop_resource8
4link_update_dense_10_biasadd_readvariableop_resource7
3link_update_dense_11_matmul_readvariableop_resource8
4link_update_dense_11_biasadd_readvariableop_resource7
3link_update_dense_12_matmul_readvariableop_resource8
4link_update_dense_12_biasadd_readvariableop_resource
identityo
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"   J   2
Reshape/shapee
ReshapeReshapeinputReshape/shape:output:0*
T0*
_output_shapes

:J2	
Reshapeq
transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose/permw
	transpose	TransposeReshape:output:0transpose/perm:output:0*
T0*
_output_shapes

:J2
	transpose}
Pad/paddingsConst*
_output_shapes

:*
dtype0*)
value B"               2
Pad/paddings`
PadPadtranspose:y:0Pad/paddings:output:0*
T0*
_output_shapes

:J2
Pad	
GatherV2/indicesConst*
_output_shapes	
:*
dtype0*ą
value§B¤"                                                      
   
   
   
   
                                                                                                                                            !   !   !   !   !   !   "   "   "   "   "   "   "   "   #   #   #   %   %   %   %   %   %   &   &   &   '   '   '   '   '   '   '   '   (   (   (   (   )   )   )   )   )   )   )   )   *   *   +   +   +   +   +   +   ,   ,   ,   ,   -   -   -   -   -   -   -   -   .   .   .   .   /   /   /   0   0   0   0   0   1   1   1   1   1   1   1   1   2   2   2   2   3   3   3   3   4   4   4   4   4   5   5   5   6   6   6   6   7   7   7   8   8   8   8   8   8   9   9   9   9   9   9   9   9   :   :   :   :   ;   ;   ;   ;   <   <   =   =   =   =   >   >   >   >   ?   ?   ?   ?   ?   @   @   A   A   A   A   A   A   B   B   B   B   C   C   C   C   C   C   C   C   D   D   D   D   E   E   E   E   F   F   F   F   G   G   G   G   H   H   H   H   H   I   I   I   I   I   I   I   I   2
GatherV2/indices`
GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2/axisŽ
GatherV2GatherV2Pad:output:0GatherV2/indices:output:0GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:	2

GatherV2	
GatherV2_1/indicesConst*
_output_shapes	
:*
dtype0*ą
value§B¤"      
                                  #   %   &   '   5   6   7   8   9          D   E   F   G   H   I                                                                         !   "                                 B   C   D   E   F   G   H   I   :   ;   <   =   >   ?   @   A         
   D   E   F   G   H   I   2   3   4   :   ;   <   =   >   ?   @   A   *   +   ,   -   :   ;   <   =   >   ?   @   A   (   )   D   E   F   G   H   I   .   /   0   1   :   ;   <   =   >   ?   @   A   *   +   ,   -   2   3   4   5   6   7   8   9   :   ;   <   =   >   ?   @   A   #   %   &   '   .   /   0   1   5   6   7   8   9         
   .   /   0   1   2   3   4   D   E   F   G   H   I   :   ;   <   =   >   ?   @   A          !   "   #   %   &   '   (   )   *   +   ,   -   .   /   0   1   5   6   7   8   9   B   C   D   E   F   G   H   I          !   "   :   ;   <   =   >   ?   @   A                      !   "   #   %   &   '   *   +   ,   -   5   6   7   8   9   :   ;   <   =   >   ?   @   A   2
GatherV2_1/indicesd
GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2_1/axisś

GatherV2_1GatherV2Pad:output:0GatherV2_1/indices:output:0GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:	2

GatherV2_1\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis
concatConcatV2GatherV2:output:0GatherV2_1:output:0concat/axis:output:0*
N*
T0*
_output_shapes
:	 2
concatŇ
,create_message/dense_8/MatMul/ReadVariableOpReadVariableOp5create_message_dense_8_matmul_readvariableop_resource*
_output_shapes

: @*
dtype02.
,create_message/dense_8/MatMul/ReadVariableOpš
create_message/dense_8/MatMulMatMulconcat:output:04create_message/dense_8/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	@2
create_message/dense_8/MatMulŃ
-create_message/dense_8/BiasAdd/ReadVariableOpReadVariableOp6create_message_dense_8_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-create_message/dense_8/BiasAdd/ReadVariableOpŐ
create_message/dense_8/BiasAddBiasAdd'create_message/dense_8/MatMul:product:05create_message/dense_8/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	@2 
create_message/dense_8/BiasAdd
create_message/dense_8/TanhTanh'create_message/dense_8/BiasAdd:output:0*
T0*
_output_shapes
:	@2
create_message/dense_8/TanhŇ
,create_message/dense_9/MatMul/ReadVariableOpReadVariableOp5create_message_dense_9_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02.
,create_message/dense_9/MatMul/ReadVariableOpÉ
create_message/dense_9/MatMulMatMulcreate_message/dense_8/Tanh:y:04create_message/dense_9/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
create_message/dense_9/MatMulŃ
-create_message/dense_9/BiasAdd/ReadVariableOpReadVariableOp6create_message_dense_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-create_message/dense_9/BiasAdd/ReadVariableOpŐ
create_message/dense_9/BiasAddBiasAdd'create_message/dense_9/MatMul:product:05create_message/dense_9/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	2 
create_message/dense_9/BiasAdd
create_message/dense_9/TanhTanh'create_message/dense_9/BiasAdd:output:0*
T0*
_output_shapes
:	2
create_message/dense_9/Tanh˛
PartitionedCallPartitionedCallcreate_message/dense_9/Tanh:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:J * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *-
f(R&
$__inference_message_aggregation_12562
PartitionedCall`
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat_1/axis
concat_1ConcatV2Pad:output:0PartitionedCall:output:0concat_1/axis:output:0*
N*
T0*
_output_shapes

:J02

concat_1Í
*link_update/dense_10/MatMul/ReadVariableOpReadVariableOp3link_update_dense_10_matmul_readvariableop_resource*
_output_shapes
:	0*
dtype02,
*link_update/dense_10/MatMul/ReadVariableOpľ
link_update/dense_10/MatMulMatMulconcat_1:output:02link_update/dense_10/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	J2
link_update/dense_10/MatMulĚ
+link_update/dense_10/BiasAdd/ReadVariableOpReadVariableOp4link_update_dense_10_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02-
+link_update/dense_10/BiasAdd/ReadVariableOpÍ
link_update/dense_10/BiasAddBiasAdd%link_update/dense_10/MatMul:product:03link_update/dense_10/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	J2
link_update/dense_10/BiasAdd
link_update/dense_10/TanhTanh%link_update/dense_10/BiasAdd:output:0*
T0*
_output_shapes
:	J2
link_update/dense_10/TanhÍ
*link_update/dense_11/MatMul/ReadVariableOpReadVariableOp3link_update_dense_11_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype02,
*link_update/dense_11/MatMul/ReadVariableOpŔ
link_update/dense_11/MatMulMatMullink_update/dense_10/Tanh:y:02link_update/dense_11/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:J@2
link_update/dense_11/MatMulË
+link_update/dense_11/BiasAdd/ReadVariableOpReadVariableOp4link_update_dense_11_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02-
+link_update/dense_11/BiasAdd/ReadVariableOpĚ
link_update/dense_11/BiasAddBiasAdd%link_update/dense_11/MatMul:product:03link_update/dense_11/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:J@2
link_update/dense_11/BiasAdd
link_update/dense_11/TanhTanh%link_update/dense_11/BiasAdd:output:0*
T0*
_output_shapes

:J@2
link_update/dense_11/TanhĚ
*link_update/dense_12/MatMul/ReadVariableOpReadVariableOp3link_update_dense_12_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02,
*link_update/dense_12/MatMul/ReadVariableOpŔ
link_update/dense_12/MatMulMatMullink_update/dense_11/Tanh:y:02link_update/dense_12/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:J2
link_update/dense_12/MatMulË
+link_update/dense_12/BiasAdd/ReadVariableOpReadVariableOp4link_update_dense_12_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02-
+link_update/dense_12/BiasAdd/ReadVariableOpĚ
link_update/dense_12/BiasAddBiasAdd%link_update/dense_12/MatMul:product:03link_update/dense_12/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:J2
link_update/dense_12/BiasAdd
link_update/dense_12/TanhTanh%link_update/dense_12/BiasAdd:output:0*
T0*
_output_shapes

:J2
link_update/dense_12/Tanh	
GatherV2_2/indicesConst*
_output_shapes	
:*
dtype0*ą
value§B¤"                                                      
   
   
   
   
                                                                                                                                            !   !   !   !   !   !   "   "   "   "   "   "   "   "   #   #   #   %   %   %   %   %   %   &   &   &   '   '   '   '   '   '   '   '   (   (   (   (   )   )   )   )   )   )   )   )   *   *   +   +   +   +   +   +   ,   ,   ,   ,   -   -   -   -   -   -   -   -   .   .   .   .   /   /   /   0   0   0   0   0   1   1   1   1   1   1   1   1   2   2   2   2   3   3   3   3   4   4   4   4   4   5   5   5   6   6   6   6   7   7   7   8   8   8   8   8   8   9   9   9   9   9   9   9   9   :   :   :   :   ;   ;   ;   ;   <   <   =   =   =   =   >   >   >   >   ?   ?   ?   ?   ?   @   @   A   A   A   A   A   A   B   B   B   B   C   C   C   C   C   C   C   C   D   D   D   D   E   E   E   E   F   F   F   F   G   G   G   G   H   H   H   H   H   I   I   I   I   I   I   I   I   2
GatherV2_2/indicesd
GatherV2_2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2_2/axisÇ

GatherV2_2GatherV2link_update/dense_12/Tanh:y:0GatherV2_2/indices:output:0GatherV2_2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:	2

GatherV2_2	
GatherV2_3/indicesConst*
_output_shapes	
:*
dtype0*ą
value§B¤"      
                                  #   %   &   '   5   6   7   8   9          D   E   F   G   H   I                                                                         !   "                                 B   C   D   E   F   G   H   I   :   ;   <   =   >   ?   @   A         
   D   E   F   G   H   I   2   3   4   :   ;   <   =   >   ?   @   A   *   +   ,   -   :   ;   <   =   >   ?   @   A   (   )   D   E   F   G   H   I   .   /   0   1   :   ;   <   =   >   ?   @   A   *   +   ,   -   2   3   4   5   6   7   8   9   :   ;   <   =   >   ?   @   A   #   %   &   '   .   /   0   1   5   6   7   8   9         
   .   /   0   1   2   3   4   D   E   F   G   H   I   :   ;   <   =   >   ?   @   A          !   "   #   %   &   '   (   )   *   +   ,   -   .   /   0   1   5   6   7   8   9   B   C   D   E   F   G   H   I          !   "   :   ;   <   =   >   ?   @   A                      !   "   #   %   &   '   *   +   ,   -   5   6   7   8   9   :   ;   <   =   >   ?   @   A   2
GatherV2_3/indicesd
GatherV2_3/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2_3/axisÇ

GatherV2_3GatherV2link_update/dense_12/Tanh:y:0GatherV2_3/indices:output:0GatherV2_3/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:	2

GatherV2_3`
concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat_2/axis
concat_2ConcatV2GatherV2_2:output:0GatherV2_3:output:0concat_2/axis:output:0*
N*
T0*
_output_shapes
:	 2

concat_2Ö
.create_message/dense_8/MatMul_1/ReadVariableOpReadVariableOp5create_message_dense_8_matmul_readvariableop_resource*
_output_shapes

: @*
dtype020
.create_message/dense_8/MatMul_1/ReadVariableOpÁ
create_message/dense_8/MatMul_1MatMulconcat_2:output:06create_message/dense_8/MatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	@2!
create_message/dense_8/MatMul_1Ő
/create_message/dense_8/BiasAdd_1/ReadVariableOpReadVariableOp6create_message_dense_8_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype021
/create_message/dense_8/BiasAdd_1/ReadVariableOpÝ
 create_message/dense_8/BiasAdd_1BiasAdd)create_message/dense_8/MatMul_1:product:07create_message/dense_8/BiasAdd_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	@2"
 create_message/dense_8/BiasAdd_1
create_message/dense_8/Tanh_1Tanh)create_message/dense_8/BiasAdd_1:output:0*
T0*
_output_shapes
:	@2
create_message/dense_8/Tanh_1Ö
.create_message/dense_9/MatMul_1/ReadVariableOpReadVariableOp5create_message_dense_9_matmul_readvariableop_resource*
_output_shapes

:@*
dtype020
.create_message/dense_9/MatMul_1/ReadVariableOpŃ
create_message/dense_9/MatMul_1MatMul!create_message/dense_8/Tanh_1:y:06create_message/dense_9/MatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	2!
create_message/dense_9/MatMul_1Ő
/create_message/dense_9/BiasAdd_1/ReadVariableOpReadVariableOp6create_message_dense_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype021
/create_message/dense_9/BiasAdd_1/ReadVariableOpÝ
 create_message/dense_9/BiasAdd_1BiasAdd)create_message/dense_9/MatMul_1:product:07create_message/dense_9/BiasAdd_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	2"
 create_message/dense_9/BiasAdd_1
create_message/dense_9/Tanh_1Tanh)create_message/dense_9/BiasAdd_1:output:0*
T0*
_output_shapes
:	2
create_message/dense_9/Tanh_1¸
PartitionedCall_1PartitionedCall!create_message/dense_9/Tanh_1:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:J * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *-
f(R&
$__inference_message_aggregation_12562
PartitionedCall_1`
concat_3/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat_3/axisĽ
concat_3ConcatV2link_update/dense_12/Tanh:y:0PartitionedCall_1:output:0concat_3/axis:output:0*
N*
T0*
_output_shapes

:J02

concat_3Ń
,link_update/dense_10/MatMul_1/ReadVariableOpReadVariableOp3link_update_dense_10_matmul_readvariableop_resource*
_output_shapes
:	0*
dtype02.
,link_update/dense_10/MatMul_1/ReadVariableOpť
link_update/dense_10/MatMul_1MatMulconcat_3:output:04link_update/dense_10/MatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	J2
link_update/dense_10/MatMul_1Đ
-link_update/dense_10/BiasAdd_1/ReadVariableOpReadVariableOp4link_update_dense_10_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02/
-link_update/dense_10/BiasAdd_1/ReadVariableOpŐ
link_update/dense_10/BiasAdd_1BiasAdd'link_update/dense_10/MatMul_1:product:05link_update/dense_10/BiasAdd_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	J2 
link_update/dense_10/BiasAdd_1
link_update/dense_10/Tanh_1Tanh'link_update/dense_10/BiasAdd_1:output:0*
T0*
_output_shapes
:	J2
link_update/dense_10/Tanh_1Ń
,link_update/dense_11/MatMul_1/ReadVariableOpReadVariableOp3link_update_dense_11_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype02.
,link_update/dense_11/MatMul_1/ReadVariableOpČ
link_update/dense_11/MatMul_1MatMullink_update/dense_10/Tanh_1:y:04link_update/dense_11/MatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes

:J@2
link_update/dense_11/MatMul_1Ď
-link_update/dense_11/BiasAdd_1/ReadVariableOpReadVariableOp4link_update_dense_11_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-link_update/dense_11/BiasAdd_1/ReadVariableOpÔ
link_update/dense_11/BiasAdd_1BiasAdd'link_update/dense_11/MatMul_1:product:05link_update/dense_11/BiasAdd_1/ReadVariableOp:value:0*
T0*
_output_shapes

:J@2 
link_update/dense_11/BiasAdd_1
link_update/dense_11/Tanh_1Tanh'link_update/dense_11/BiasAdd_1:output:0*
T0*
_output_shapes

:J@2
link_update/dense_11/Tanh_1Đ
,link_update/dense_12/MatMul_1/ReadVariableOpReadVariableOp3link_update_dense_12_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02.
,link_update/dense_12/MatMul_1/ReadVariableOpČ
link_update/dense_12/MatMul_1MatMullink_update/dense_11/Tanh_1:y:04link_update/dense_12/MatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes

:J2
link_update/dense_12/MatMul_1Ď
-link_update/dense_12/BiasAdd_1/ReadVariableOpReadVariableOp4link_update_dense_12_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-link_update/dense_12/BiasAdd_1/ReadVariableOpÔ
link_update/dense_12/BiasAdd_1BiasAdd'link_update/dense_12/MatMul_1:product:05link_update/dense_12/BiasAdd_1/ReadVariableOp:value:0*
T0*
_output_shapes

:J2 
link_update/dense_12/BiasAdd_1
link_update/dense_12/Tanh_1Tanh'link_update/dense_12/BiasAdd_1:output:0*
T0*
_output_shapes

:J2
link_update/dense_12/Tanh_1	
GatherV2_4/indicesConst*
_output_shapes	
:*
dtype0*ą
value§B¤"                                                      
   
   
   
   
                                                                                                                                            !   !   !   !   !   !   "   "   "   "   "   "   "   "   #   #   #   %   %   %   %   %   %   &   &   &   '   '   '   '   '   '   '   '   (   (   (   (   )   )   )   )   )   )   )   )   *   *   +   +   +   +   +   +   ,   ,   ,   ,   -   -   -   -   -   -   -   -   .   .   .   .   /   /   /   0   0   0   0   0   1   1   1   1   1   1   1   1   2   2   2   2   3   3   3   3   4   4   4   4   4   5   5   5   6   6   6   6   7   7   7   8   8   8   8   8   8   9   9   9   9   9   9   9   9   :   :   :   :   ;   ;   ;   ;   <   <   =   =   =   =   >   >   >   >   ?   ?   ?   ?   ?   @   @   A   A   A   A   A   A   B   B   B   B   C   C   C   C   C   C   C   C   D   D   D   D   E   E   E   E   F   F   F   F   G   G   G   G   H   H   H   H   H   I   I   I   I   I   I   I   I   2
GatherV2_4/indicesd
GatherV2_4/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2_4/axisÉ

GatherV2_4GatherV2link_update/dense_12/Tanh_1:y:0GatherV2_4/indices:output:0GatherV2_4/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:	2

GatherV2_4	
GatherV2_5/indicesConst*
_output_shapes	
:*
dtype0*ą
value§B¤"      
                                  #   %   &   '   5   6   7   8   9          D   E   F   G   H   I                                                                         !   "                                 B   C   D   E   F   G   H   I   :   ;   <   =   >   ?   @   A         
   D   E   F   G   H   I   2   3   4   :   ;   <   =   >   ?   @   A   *   +   ,   -   :   ;   <   =   >   ?   @   A   (   )   D   E   F   G   H   I   .   /   0   1   :   ;   <   =   >   ?   @   A   *   +   ,   -   2   3   4   5   6   7   8   9   :   ;   <   =   >   ?   @   A   #   %   &   '   .   /   0   1   5   6   7   8   9         
   .   /   0   1   2   3   4   D   E   F   G   H   I   :   ;   <   =   >   ?   @   A          !   "   #   %   &   '   (   )   *   +   ,   -   .   /   0   1   5   6   7   8   9   B   C   D   E   F   G   H   I          !   "   :   ;   <   =   >   ?   @   A                      !   "   #   %   &   '   *   +   ,   -   5   6   7   8   9   :   ;   <   =   >   ?   @   A   2
GatherV2_5/indicesd
GatherV2_5/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2_5/axisÉ

GatherV2_5GatherV2link_update/dense_12/Tanh_1:y:0GatherV2_5/indices:output:0GatherV2_5/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:	2

GatherV2_5`
concat_4/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat_4/axis
concat_4ConcatV2GatherV2_4:output:0GatherV2_5:output:0concat_4/axis:output:0*
N*
T0*
_output_shapes
:	 2

concat_4Ö
.create_message/dense_8/MatMul_2/ReadVariableOpReadVariableOp5create_message_dense_8_matmul_readvariableop_resource*
_output_shapes

: @*
dtype020
.create_message/dense_8/MatMul_2/ReadVariableOpÁ
create_message/dense_8/MatMul_2MatMulconcat_4:output:06create_message/dense_8/MatMul_2/ReadVariableOp:value:0*
T0*
_output_shapes
:	@2!
create_message/dense_8/MatMul_2Ő
/create_message/dense_8/BiasAdd_2/ReadVariableOpReadVariableOp6create_message_dense_8_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype021
/create_message/dense_8/BiasAdd_2/ReadVariableOpÝ
 create_message/dense_8/BiasAdd_2BiasAdd)create_message/dense_8/MatMul_2:product:07create_message/dense_8/BiasAdd_2/ReadVariableOp:value:0*
T0*
_output_shapes
:	@2"
 create_message/dense_8/BiasAdd_2
create_message/dense_8/Tanh_2Tanh)create_message/dense_8/BiasAdd_2:output:0*
T0*
_output_shapes
:	@2
create_message/dense_8/Tanh_2Ö
.create_message/dense_9/MatMul_2/ReadVariableOpReadVariableOp5create_message_dense_9_matmul_readvariableop_resource*
_output_shapes

:@*
dtype020
.create_message/dense_9/MatMul_2/ReadVariableOpŃ
create_message/dense_9/MatMul_2MatMul!create_message/dense_8/Tanh_2:y:06create_message/dense_9/MatMul_2/ReadVariableOp:value:0*
T0*
_output_shapes
:	2!
create_message/dense_9/MatMul_2Ő
/create_message/dense_9/BiasAdd_2/ReadVariableOpReadVariableOp6create_message_dense_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype021
/create_message/dense_9/BiasAdd_2/ReadVariableOpÝ
 create_message/dense_9/BiasAdd_2BiasAdd)create_message/dense_9/MatMul_2:product:07create_message/dense_9/BiasAdd_2/ReadVariableOp:value:0*
T0*
_output_shapes
:	2"
 create_message/dense_9/BiasAdd_2
create_message/dense_9/Tanh_2Tanh)create_message/dense_9/BiasAdd_2:output:0*
T0*
_output_shapes
:	2
create_message/dense_9/Tanh_2¸
PartitionedCall_2PartitionedCall!create_message/dense_9/Tanh_2:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:J * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *-
f(R&
$__inference_message_aggregation_12562
PartitionedCall_2`
concat_5/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat_5/axis§
concat_5ConcatV2link_update/dense_12/Tanh_1:y:0PartitionedCall_2:output:0concat_5/axis:output:0*
N*
T0*
_output_shapes

:J02

concat_5Ń
,link_update/dense_10/MatMul_2/ReadVariableOpReadVariableOp3link_update_dense_10_matmul_readvariableop_resource*
_output_shapes
:	0*
dtype02.
,link_update/dense_10/MatMul_2/ReadVariableOpť
link_update/dense_10/MatMul_2MatMulconcat_5:output:04link_update/dense_10/MatMul_2/ReadVariableOp:value:0*
T0*
_output_shapes
:	J2
link_update/dense_10/MatMul_2Đ
-link_update/dense_10/BiasAdd_2/ReadVariableOpReadVariableOp4link_update_dense_10_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02/
-link_update/dense_10/BiasAdd_2/ReadVariableOpŐ
link_update/dense_10/BiasAdd_2BiasAdd'link_update/dense_10/MatMul_2:product:05link_update/dense_10/BiasAdd_2/ReadVariableOp:value:0*
T0*
_output_shapes
:	J2 
link_update/dense_10/BiasAdd_2
link_update/dense_10/Tanh_2Tanh'link_update/dense_10/BiasAdd_2:output:0*
T0*
_output_shapes
:	J2
link_update/dense_10/Tanh_2Ń
,link_update/dense_11/MatMul_2/ReadVariableOpReadVariableOp3link_update_dense_11_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype02.
,link_update/dense_11/MatMul_2/ReadVariableOpČ
link_update/dense_11/MatMul_2MatMullink_update/dense_10/Tanh_2:y:04link_update/dense_11/MatMul_2/ReadVariableOp:value:0*
T0*
_output_shapes

:J@2
link_update/dense_11/MatMul_2Ď
-link_update/dense_11/BiasAdd_2/ReadVariableOpReadVariableOp4link_update_dense_11_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-link_update/dense_11/BiasAdd_2/ReadVariableOpÔ
link_update/dense_11/BiasAdd_2BiasAdd'link_update/dense_11/MatMul_2:product:05link_update/dense_11/BiasAdd_2/ReadVariableOp:value:0*
T0*
_output_shapes

:J@2 
link_update/dense_11/BiasAdd_2
link_update/dense_11/Tanh_2Tanh'link_update/dense_11/BiasAdd_2:output:0*
T0*
_output_shapes

:J@2
link_update/dense_11/Tanh_2Đ
,link_update/dense_12/MatMul_2/ReadVariableOpReadVariableOp3link_update_dense_12_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02.
,link_update/dense_12/MatMul_2/ReadVariableOpČ
link_update/dense_12/MatMul_2MatMullink_update/dense_11/Tanh_2:y:04link_update/dense_12/MatMul_2/ReadVariableOp:value:0*
T0*
_output_shapes

:J2
link_update/dense_12/MatMul_2Ď
-link_update/dense_12/BiasAdd_2/ReadVariableOpReadVariableOp4link_update_dense_12_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-link_update/dense_12/BiasAdd_2/ReadVariableOpÔ
link_update/dense_12/BiasAdd_2BiasAdd'link_update/dense_12/MatMul_2:product:05link_update/dense_12/BiasAdd_2/ReadVariableOp:value:0*
T0*
_output_shapes

:J2 
link_update/dense_12/BiasAdd_2
link_update/dense_12/Tanh_2Tanh'link_update/dense_12/BiasAdd_2:output:0*
T0*
_output_shapes

:J2
link_update/dense_12/Tanh_2	
GatherV2_6/indicesConst*
_output_shapes	
:*
dtype0*ą
value§B¤"                                                      
   
   
   
   
                                                                                                                                            !   !   !   !   !   !   "   "   "   "   "   "   "   "   #   #   #   %   %   %   %   %   %   &   &   &   '   '   '   '   '   '   '   '   (   (   (   (   )   )   )   )   )   )   )   )   *   *   +   +   +   +   +   +   ,   ,   ,   ,   -   -   -   -   -   -   -   -   .   .   .   .   /   /   /   0   0   0   0   0   1   1   1   1   1   1   1   1   2   2   2   2   3   3   3   3   4   4   4   4   4   5   5   5   6   6   6   6   7   7   7   8   8   8   8   8   8   9   9   9   9   9   9   9   9   :   :   :   :   ;   ;   ;   ;   <   <   =   =   =   =   >   >   >   >   ?   ?   ?   ?   ?   @   @   A   A   A   A   A   A   B   B   B   B   C   C   C   C   C   C   C   C   D   D   D   D   E   E   E   E   F   F   F   F   G   G   G   G   H   H   H   H   H   I   I   I   I   I   I   I   I   2
GatherV2_6/indicesd
GatherV2_6/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2_6/axisÉ

GatherV2_6GatherV2link_update/dense_12/Tanh_2:y:0GatherV2_6/indices:output:0GatherV2_6/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:	2

GatherV2_6	
GatherV2_7/indicesConst*
_output_shapes	
:*
dtype0*ą
value§B¤"      
                                  #   %   &   '   5   6   7   8   9          D   E   F   G   H   I                                                                         !   "                                 B   C   D   E   F   G   H   I   :   ;   <   =   >   ?   @   A         
   D   E   F   G   H   I   2   3   4   :   ;   <   =   >   ?   @   A   *   +   ,   -   :   ;   <   =   >   ?   @   A   (   )   D   E   F   G   H   I   .   /   0   1   :   ;   <   =   >   ?   @   A   *   +   ,   -   2   3   4   5   6   7   8   9   :   ;   <   =   >   ?   @   A   #   %   &   '   .   /   0   1   5   6   7   8   9         
   .   /   0   1   2   3   4   D   E   F   G   H   I   :   ;   <   =   >   ?   @   A          !   "   #   %   &   '   (   )   *   +   ,   -   .   /   0   1   5   6   7   8   9   B   C   D   E   F   G   H   I          !   "   :   ;   <   =   >   ?   @   A                      !   "   #   %   &   '   *   +   ,   -   5   6   7   8   9   :   ;   <   =   >   ?   @   A   2
GatherV2_7/indicesd
GatherV2_7/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2_7/axisÉ

GatherV2_7GatherV2link_update/dense_12/Tanh_2:y:0GatherV2_7/indices:output:0GatherV2_7/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:	2

GatherV2_7`
concat_6/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat_6/axis
concat_6ConcatV2GatherV2_6:output:0GatherV2_7:output:0concat_6/axis:output:0*
N*
T0*
_output_shapes
:	 2

concat_6Ö
.create_message/dense_8/MatMul_3/ReadVariableOpReadVariableOp5create_message_dense_8_matmul_readvariableop_resource*
_output_shapes

: @*
dtype020
.create_message/dense_8/MatMul_3/ReadVariableOpÁ
create_message/dense_8/MatMul_3MatMulconcat_6:output:06create_message/dense_8/MatMul_3/ReadVariableOp:value:0*
T0*
_output_shapes
:	@2!
create_message/dense_8/MatMul_3Ő
/create_message/dense_8/BiasAdd_3/ReadVariableOpReadVariableOp6create_message_dense_8_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype021
/create_message/dense_8/BiasAdd_3/ReadVariableOpÝ
 create_message/dense_8/BiasAdd_3BiasAdd)create_message/dense_8/MatMul_3:product:07create_message/dense_8/BiasAdd_3/ReadVariableOp:value:0*
T0*
_output_shapes
:	@2"
 create_message/dense_8/BiasAdd_3
create_message/dense_8/Tanh_3Tanh)create_message/dense_8/BiasAdd_3:output:0*
T0*
_output_shapes
:	@2
create_message/dense_8/Tanh_3Ö
.create_message/dense_9/MatMul_3/ReadVariableOpReadVariableOp5create_message_dense_9_matmul_readvariableop_resource*
_output_shapes

:@*
dtype020
.create_message/dense_9/MatMul_3/ReadVariableOpŃ
create_message/dense_9/MatMul_3MatMul!create_message/dense_8/Tanh_3:y:06create_message/dense_9/MatMul_3/ReadVariableOp:value:0*
T0*
_output_shapes
:	2!
create_message/dense_9/MatMul_3Ő
/create_message/dense_9/BiasAdd_3/ReadVariableOpReadVariableOp6create_message_dense_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype021
/create_message/dense_9/BiasAdd_3/ReadVariableOpÝ
 create_message/dense_9/BiasAdd_3BiasAdd)create_message/dense_9/MatMul_3:product:07create_message/dense_9/BiasAdd_3/ReadVariableOp:value:0*
T0*
_output_shapes
:	2"
 create_message/dense_9/BiasAdd_3
create_message/dense_9/Tanh_3Tanh)create_message/dense_9/BiasAdd_3:output:0*
T0*
_output_shapes
:	2
create_message/dense_9/Tanh_3¸
PartitionedCall_3PartitionedCall!create_message/dense_9/Tanh_3:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:J * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *-
f(R&
$__inference_message_aggregation_12562
PartitionedCall_3`
concat_7/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat_7/axis§
concat_7ConcatV2link_update/dense_12/Tanh_2:y:0PartitionedCall_3:output:0concat_7/axis:output:0*
N*
T0*
_output_shapes

:J02

concat_7Ń
,link_update/dense_10/MatMul_3/ReadVariableOpReadVariableOp3link_update_dense_10_matmul_readvariableop_resource*
_output_shapes
:	0*
dtype02.
,link_update/dense_10/MatMul_3/ReadVariableOpť
link_update/dense_10/MatMul_3MatMulconcat_7:output:04link_update/dense_10/MatMul_3/ReadVariableOp:value:0*
T0*
_output_shapes
:	J2
link_update/dense_10/MatMul_3Đ
-link_update/dense_10/BiasAdd_3/ReadVariableOpReadVariableOp4link_update_dense_10_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02/
-link_update/dense_10/BiasAdd_3/ReadVariableOpŐ
link_update/dense_10/BiasAdd_3BiasAdd'link_update/dense_10/MatMul_3:product:05link_update/dense_10/BiasAdd_3/ReadVariableOp:value:0*
T0*
_output_shapes
:	J2 
link_update/dense_10/BiasAdd_3
link_update/dense_10/Tanh_3Tanh'link_update/dense_10/BiasAdd_3:output:0*
T0*
_output_shapes
:	J2
link_update/dense_10/Tanh_3Ń
,link_update/dense_11/MatMul_3/ReadVariableOpReadVariableOp3link_update_dense_11_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype02.
,link_update/dense_11/MatMul_3/ReadVariableOpČ
link_update/dense_11/MatMul_3MatMullink_update/dense_10/Tanh_3:y:04link_update/dense_11/MatMul_3/ReadVariableOp:value:0*
T0*
_output_shapes

:J@2
link_update/dense_11/MatMul_3Ď
-link_update/dense_11/BiasAdd_3/ReadVariableOpReadVariableOp4link_update_dense_11_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-link_update/dense_11/BiasAdd_3/ReadVariableOpÔ
link_update/dense_11/BiasAdd_3BiasAdd'link_update/dense_11/MatMul_3:product:05link_update/dense_11/BiasAdd_3/ReadVariableOp:value:0*
T0*
_output_shapes

:J@2 
link_update/dense_11/BiasAdd_3
link_update/dense_11/Tanh_3Tanh'link_update/dense_11/BiasAdd_3:output:0*
T0*
_output_shapes

:J@2
link_update/dense_11/Tanh_3Đ
,link_update/dense_12/MatMul_3/ReadVariableOpReadVariableOp3link_update_dense_12_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02.
,link_update/dense_12/MatMul_3/ReadVariableOpČ
link_update/dense_12/MatMul_3MatMullink_update/dense_11/Tanh_3:y:04link_update/dense_12/MatMul_3/ReadVariableOp:value:0*
T0*
_output_shapes

:J2
link_update/dense_12/MatMul_3Ď
-link_update/dense_12/BiasAdd_3/ReadVariableOpReadVariableOp4link_update_dense_12_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-link_update/dense_12/BiasAdd_3/ReadVariableOpÔ
link_update/dense_12/BiasAdd_3BiasAdd'link_update/dense_12/MatMul_3:product:05link_update/dense_12/BiasAdd_3/ReadVariableOp:value:0*
T0*
_output_shapes

:J2 
link_update/dense_12/BiasAdd_3
link_update/dense_12/Tanh_3Tanh'link_update/dense_12/BiasAdd_3:output:0*
T0*
_output_shapes

:J2
link_update/dense_12/Tanh_3	
GatherV2_8/indicesConst*
_output_shapes	
:*
dtype0*ą
value§B¤"                                                      
   
   
   
   
                                                                                                                                            !   !   !   !   !   !   "   "   "   "   "   "   "   "   #   #   #   %   %   %   %   %   %   &   &   &   '   '   '   '   '   '   '   '   (   (   (   (   )   )   )   )   )   )   )   )   *   *   +   +   +   +   +   +   ,   ,   ,   ,   -   -   -   -   -   -   -   -   .   .   .   .   /   /   /   0   0   0   0   0   1   1   1   1   1   1   1   1   2   2   2   2   3   3   3   3   4   4   4   4   4   5   5   5   6   6   6   6   7   7   7   8   8   8   8   8   8   9   9   9   9   9   9   9   9   :   :   :   :   ;   ;   ;   ;   <   <   =   =   =   =   >   >   >   >   ?   ?   ?   ?   ?   @   @   A   A   A   A   A   A   B   B   B   B   C   C   C   C   C   C   C   C   D   D   D   D   E   E   E   E   F   F   F   F   G   G   G   G   H   H   H   H   H   I   I   I   I   I   I   I   I   2
GatherV2_8/indicesd
GatherV2_8/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2_8/axisÉ

GatherV2_8GatherV2link_update/dense_12/Tanh_3:y:0GatherV2_8/indices:output:0GatherV2_8/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:	2

GatherV2_8	
GatherV2_9/indicesConst*
_output_shapes	
:*
dtype0*ą
value§B¤"      
                                  #   %   &   '   5   6   7   8   9          D   E   F   G   H   I                                                                         !   "                                 B   C   D   E   F   G   H   I   :   ;   <   =   >   ?   @   A         
   D   E   F   G   H   I   2   3   4   :   ;   <   =   >   ?   @   A   *   +   ,   -   :   ;   <   =   >   ?   @   A   (   )   D   E   F   G   H   I   .   /   0   1   :   ;   <   =   >   ?   @   A   *   +   ,   -   2   3   4   5   6   7   8   9   :   ;   <   =   >   ?   @   A   #   %   &   '   .   /   0   1   5   6   7   8   9         
   .   /   0   1   2   3   4   D   E   F   G   H   I   :   ;   <   =   >   ?   @   A          !   "   #   %   &   '   (   )   *   +   ,   -   .   /   0   1   5   6   7   8   9   B   C   D   E   F   G   H   I          !   "   :   ;   <   =   >   ?   @   A                      !   "   #   %   &   '   *   +   ,   -   5   6   7   8   9   :   ;   <   =   >   ?   @   A   2
GatherV2_9/indicesd
GatherV2_9/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2_9/axisÉ

GatherV2_9GatherV2link_update/dense_12/Tanh_3:y:0GatherV2_9/indices:output:0GatherV2_9/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:	2

GatherV2_9`
concat_8/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat_8/axis
concat_8ConcatV2GatherV2_8:output:0GatherV2_9:output:0concat_8/axis:output:0*
N*
T0*
_output_shapes
:	 2

concat_8Ö
.create_message/dense_8/MatMul_4/ReadVariableOpReadVariableOp5create_message_dense_8_matmul_readvariableop_resource*
_output_shapes

: @*
dtype020
.create_message/dense_8/MatMul_4/ReadVariableOpÁ
create_message/dense_8/MatMul_4MatMulconcat_8:output:06create_message/dense_8/MatMul_4/ReadVariableOp:value:0*
T0*
_output_shapes
:	@2!
create_message/dense_8/MatMul_4Ő
/create_message/dense_8/BiasAdd_4/ReadVariableOpReadVariableOp6create_message_dense_8_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype021
/create_message/dense_8/BiasAdd_4/ReadVariableOpÝ
 create_message/dense_8/BiasAdd_4BiasAdd)create_message/dense_8/MatMul_4:product:07create_message/dense_8/BiasAdd_4/ReadVariableOp:value:0*
T0*
_output_shapes
:	@2"
 create_message/dense_8/BiasAdd_4
create_message/dense_8/Tanh_4Tanh)create_message/dense_8/BiasAdd_4:output:0*
T0*
_output_shapes
:	@2
create_message/dense_8/Tanh_4Ö
.create_message/dense_9/MatMul_4/ReadVariableOpReadVariableOp5create_message_dense_9_matmul_readvariableop_resource*
_output_shapes

:@*
dtype020
.create_message/dense_9/MatMul_4/ReadVariableOpŃ
create_message/dense_9/MatMul_4MatMul!create_message/dense_8/Tanh_4:y:06create_message/dense_9/MatMul_4/ReadVariableOp:value:0*
T0*
_output_shapes
:	2!
create_message/dense_9/MatMul_4Ő
/create_message/dense_9/BiasAdd_4/ReadVariableOpReadVariableOp6create_message_dense_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype021
/create_message/dense_9/BiasAdd_4/ReadVariableOpÝ
 create_message/dense_9/BiasAdd_4BiasAdd)create_message/dense_9/MatMul_4:product:07create_message/dense_9/BiasAdd_4/ReadVariableOp:value:0*
T0*
_output_shapes
:	2"
 create_message/dense_9/BiasAdd_4
create_message/dense_9/Tanh_4Tanh)create_message/dense_9/BiasAdd_4:output:0*
T0*
_output_shapes
:	2
create_message/dense_9/Tanh_4¸
PartitionedCall_4PartitionedCall!create_message/dense_9/Tanh_4:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:J * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *-
f(R&
$__inference_message_aggregation_12562
PartitionedCall_4`
concat_9/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat_9/axis§
concat_9ConcatV2link_update/dense_12/Tanh_3:y:0PartitionedCall_4:output:0concat_9/axis:output:0*
N*
T0*
_output_shapes

:J02

concat_9Ń
,link_update/dense_10/MatMul_4/ReadVariableOpReadVariableOp3link_update_dense_10_matmul_readvariableop_resource*
_output_shapes
:	0*
dtype02.
,link_update/dense_10/MatMul_4/ReadVariableOpť
link_update/dense_10/MatMul_4MatMulconcat_9:output:04link_update/dense_10/MatMul_4/ReadVariableOp:value:0*
T0*
_output_shapes
:	J2
link_update/dense_10/MatMul_4Đ
-link_update/dense_10/BiasAdd_4/ReadVariableOpReadVariableOp4link_update_dense_10_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02/
-link_update/dense_10/BiasAdd_4/ReadVariableOpŐ
link_update/dense_10/BiasAdd_4BiasAdd'link_update/dense_10/MatMul_4:product:05link_update/dense_10/BiasAdd_4/ReadVariableOp:value:0*
T0*
_output_shapes
:	J2 
link_update/dense_10/BiasAdd_4
link_update/dense_10/Tanh_4Tanh'link_update/dense_10/BiasAdd_4:output:0*
T0*
_output_shapes
:	J2
link_update/dense_10/Tanh_4Ń
,link_update/dense_11/MatMul_4/ReadVariableOpReadVariableOp3link_update_dense_11_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype02.
,link_update/dense_11/MatMul_4/ReadVariableOpČ
link_update/dense_11/MatMul_4MatMullink_update/dense_10/Tanh_4:y:04link_update/dense_11/MatMul_4/ReadVariableOp:value:0*
T0*
_output_shapes

:J@2
link_update/dense_11/MatMul_4Ď
-link_update/dense_11/BiasAdd_4/ReadVariableOpReadVariableOp4link_update_dense_11_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-link_update/dense_11/BiasAdd_4/ReadVariableOpÔ
link_update/dense_11/BiasAdd_4BiasAdd'link_update/dense_11/MatMul_4:product:05link_update/dense_11/BiasAdd_4/ReadVariableOp:value:0*
T0*
_output_shapes

:J@2 
link_update/dense_11/BiasAdd_4
link_update/dense_11/Tanh_4Tanh'link_update/dense_11/BiasAdd_4:output:0*
T0*
_output_shapes

:J@2
link_update/dense_11/Tanh_4Đ
,link_update/dense_12/MatMul_4/ReadVariableOpReadVariableOp3link_update_dense_12_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02.
,link_update/dense_12/MatMul_4/ReadVariableOpČ
link_update/dense_12/MatMul_4MatMullink_update/dense_11/Tanh_4:y:04link_update/dense_12/MatMul_4/ReadVariableOp:value:0*
T0*
_output_shapes

:J2
link_update/dense_12/MatMul_4Ď
-link_update/dense_12/BiasAdd_4/ReadVariableOpReadVariableOp4link_update_dense_12_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-link_update/dense_12/BiasAdd_4/ReadVariableOpÔ
link_update/dense_12/BiasAdd_4BiasAdd'link_update/dense_12/MatMul_4:product:05link_update/dense_12/BiasAdd_4/ReadVariableOp:value:0*
T0*
_output_shapes

:J2 
link_update/dense_12/BiasAdd_4
link_update/dense_12/Tanh_4Tanh'link_update/dense_12/BiasAdd_4:output:0*
T0*
_output_shapes

:J2
link_update/dense_12/Tanh_4	
GatherV2_10/indicesConst*
_output_shapes	
:*
dtype0*ą
value§B¤"                                                      
   
   
   
   
                                                                                                                                            !   !   !   !   !   !   "   "   "   "   "   "   "   "   #   #   #   %   %   %   %   %   %   &   &   &   '   '   '   '   '   '   '   '   (   (   (   (   )   )   )   )   )   )   )   )   *   *   +   +   +   +   +   +   ,   ,   ,   ,   -   -   -   -   -   -   -   -   .   .   .   .   /   /   /   0   0   0   0   0   1   1   1   1   1   1   1   1   2   2   2   2   3   3   3   3   4   4   4   4   4   5   5   5   6   6   6   6   7   7   7   8   8   8   8   8   8   9   9   9   9   9   9   9   9   :   :   :   :   ;   ;   ;   ;   <   <   =   =   =   =   >   >   >   >   ?   ?   ?   ?   ?   @   @   A   A   A   A   A   A   B   B   B   B   C   C   C   C   C   C   C   C   D   D   D   D   E   E   E   E   F   F   F   F   G   G   G   G   H   H   H   H   H   I   I   I   I   I   I   I   I   2
GatherV2_10/indicesf
GatherV2_10/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2_10/axisÍ
GatherV2_10GatherV2link_update/dense_12/Tanh_4:y:0GatherV2_10/indices:output:0GatherV2_10/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:	2
GatherV2_10	
GatherV2_11/indicesConst*
_output_shapes	
:*
dtype0*ą
value§B¤"      
                                  #   %   &   '   5   6   7   8   9          D   E   F   G   H   I                                                                         !   "                                 B   C   D   E   F   G   H   I   :   ;   <   =   >   ?   @   A         
   D   E   F   G   H   I   2   3   4   :   ;   <   =   >   ?   @   A   *   +   ,   -   :   ;   <   =   >   ?   @   A   (   )   D   E   F   G   H   I   .   /   0   1   :   ;   <   =   >   ?   @   A   *   +   ,   -   2   3   4   5   6   7   8   9   :   ;   <   =   >   ?   @   A   #   %   &   '   .   /   0   1   5   6   7   8   9         
   .   /   0   1   2   3   4   D   E   F   G   H   I   :   ;   <   =   >   ?   @   A          !   "   #   %   &   '   (   )   *   +   ,   -   .   /   0   1   5   6   7   8   9   B   C   D   E   F   G   H   I          !   "   :   ;   <   =   >   ?   @   A                      !   "   #   %   &   '   *   +   ,   -   5   6   7   8   9   :   ;   <   =   >   ?   @   A   2
GatherV2_11/indicesf
GatherV2_11/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2_11/axisÍ
GatherV2_11GatherV2link_update/dense_12/Tanh_4:y:0GatherV2_11/indices:output:0GatherV2_11/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:	2
GatherV2_11b
concat_10/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat_10/axis
	concat_10ConcatV2GatherV2_10:output:0GatherV2_11:output:0concat_10/axis:output:0*
N*
T0*
_output_shapes
:	 2
	concat_10Ö
.create_message/dense_8/MatMul_5/ReadVariableOpReadVariableOp5create_message_dense_8_matmul_readvariableop_resource*
_output_shapes

: @*
dtype020
.create_message/dense_8/MatMul_5/ReadVariableOpÂ
create_message/dense_8/MatMul_5MatMulconcat_10:output:06create_message/dense_8/MatMul_5/ReadVariableOp:value:0*
T0*
_output_shapes
:	@2!
create_message/dense_8/MatMul_5Ő
/create_message/dense_8/BiasAdd_5/ReadVariableOpReadVariableOp6create_message_dense_8_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype021
/create_message/dense_8/BiasAdd_5/ReadVariableOpÝ
 create_message/dense_8/BiasAdd_5BiasAdd)create_message/dense_8/MatMul_5:product:07create_message/dense_8/BiasAdd_5/ReadVariableOp:value:0*
T0*
_output_shapes
:	@2"
 create_message/dense_8/BiasAdd_5
create_message/dense_8/Tanh_5Tanh)create_message/dense_8/BiasAdd_5:output:0*
T0*
_output_shapes
:	@2
create_message/dense_8/Tanh_5Ö
.create_message/dense_9/MatMul_5/ReadVariableOpReadVariableOp5create_message_dense_9_matmul_readvariableop_resource*
_output_shapes

:@*
dtype020
.create_message/dense_9/MatMul_5/ReadVariableOpŃ
create_message/dense_9/MatMul_5MatMul!create_message/dense_8/Tanh_5:y:06create_message/dense_9/MatMul_5/ReadVariableOp:value:0*
T0*
_output_shapes
:	2!
create_message/dense_9/MatMul_5Ő
/create_message/dense_9/BiasAdd_5/ReadVariableOpReadVariableOp6create_message_dense_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype021
/create_message/dense_9/BiasAdd_5/ReadVariableOpÝ
 create_message/dense_9/BiasAdd_5BiasAdd)create_message/dense_9/MatMul_5:product:07create_message/dense_9/BiasAdd_5/ReadVariableOp:value:0*
T0*
_output_shapes
:	2"
 create_message/dense_9/BiasAdd_5
create_message/dense_9/Tanh_5Tanh)create_message/dense_9/BiasAdd_5:output:0*
T0*
_output_shapes
:	2
create_message/dense_9/Tanh_5¸
PartitionedCall_5PartitionedCall!create_message/dense_9/Tanh_5:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:J * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *-
f(R&
$__inference_message_aggregation_12562
PartitionedCall_5b
concat_11/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat_11/axisŞ
	concat_11ConcatV2link_update/dense_12/Tanh_4:y:0PartitionedCall_5:output:0concat_11/axis:output:0*
N*
T0*
_output_shapes

:J02
	concat_11Ń
,link_update/dense_10/MatMul_5/ReadVariableOpReadVariableOp3link_update_dense_10_matmul_readvariableop_resource*
_output_shapes
:	0*
dtype02.
,link_update/dense_10/MatMul_5/ReadVariableOpź
link_update/dense_10/MatMul_5MatMulconcat_11:output:04link_update/dense_10/MatMul_5/ReadVariableOp:value:0*
T0*
_output_shapes
:	J2
link_update/dense_10/MatMul_5Đ
-link_update/dense_10/BiasAdd_5/ReadVariableOpReadVariableOp4link_update_dense_10_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02/
-link_update/dense_10/BiasAdd_5/ReadVariableOpŐ
link_update/dense_10/BiasAdd_5BiasAdd'link_update/dense_10/MatMul_5:product:05link_update/dense_10/BiasAdd_5/ReadVariableOp:value:0*
T0*
_output_shapes
:	J2 
link_update/dense_10/BiasAdd_5
link_update/dense_10/Tanh_5Tanh'link_update/dense_10/BiasAdd_5:output:0*
T0*
_output_shapes
:	J2
link_update/dense_10/Tanh_5Ń
,link_update/dense_11/MatMul_5/ReadVariableOpReadVariableOp3link_update_dense_11_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype02.
,link_update/dense_11/MatMul_5/ReadVariableOpČ
link_update/dense_11/MatMul_5MatMullink_update/dense_10/Tanh_5:y:04link_update/dense_11/MatMul_5/ReadVariableOp:value:0*
T0*
_output_shapes

:J@2
link_update/dense_11/MatMul_5Ď
-link_update/dense_11/BiasAdd_5/ReadVariableOpReadVariableOp4link_update_dense_11_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-link_update/dense_11/BiasAdd_5/ReadVariableOpÔ
link_update/dense_11/BiasAdd_5BiasAdd'link_update/dense_11/MatMul_5:product:05link_update/dense_11/BiasAdd_5/ReadVariableOp:value:0*
T0*
_output_shapes

:J@2 
link_update/dense_11/BiasAdd_5
link_update/dense_11/Tanh_5Tanh'link_update/dense_11/BiasAdd_5:output:0*
T0*
_output_shapes

:J@2
link_update/dense_11/Tanh_5Đ
,link_update/dense_12/MatMul_5/ReadVariableOpReadVariableOp3link_update_dense_12_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02.
,link_update/dense_12/MatMul_5/ReadVariableOpČ
link_update/dense_12/MatMul_5MatMullink_update/dense_11/Tanh_5:y:04link_update/dense_12/MatMul_5/ReadVariableOp:value:0*
T0*
_output_shapes

:J2
link_update/dense_12/MatMul_5Ď
-link_update/dense_12/BiasAdd_5/ReadVariableOpReadVariableOp4link_update_dense_12_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-link_update/dense_12/BiasAdd_5/ReadVariableOpÔ
link_update/dense_12/BiasAdd_5BiasAdd'link_update/dense_12/MatMul_5:product:05link_update/dense_12/BiasAdd_5/ReadVariableOp:value:0*
T0*
_output_shapes

:J2 
link_update/dense_12/BiasAdd_5
link_update/dense_12/Tanh_5Tanh'link_update/dense_12/BiasAdd_5:output:0*
T0*
_output_shapes

:J2
link_update/dense_12/Tanh_5	
GatherV2_12/indicesConst*
_output_shapes	
:*
dtype0*ą
value§B¤"                                                      
   
   
   
   
                                                                                                                                            !   !   !   !   !   !   "   "   "   "   "   "   "   "   #   #   #   %   %   %   %   %   %   &   &   &   '   '   '   '   '   '   '   '   (   (   (   (   )   )   )   )   )   )   )   )   *   *   +   +   +   +   +   +   ,   ,   ,   ,   -   -   -   -   -   -   -   -   .   .   .   .   /   /   /   0   0   0   0   0   1   1   1   1   1   1   1   1   2   2   2   2   3   3   3   3   4   4   4   4   4   5   5   5   6   6   6   6   7   7   7   8   8   8   8   8   8   9   9   9   9   9   9   9   9   :   :   :   :   ;   ;   ;   ;   <   <   =   =   =   =   >   >   >   >   ?   ?   ?   ?   ?   @   @   A   A   A   A   A   A   B   B   B   B   C   C   C   C   C   C   C   C   D   D   D   D   E   E   E   E   F   F   F   F   G   G   G   G   H   H   H   H   H   I   I   I   I   I   I   I   I   2
GatherV2_12/indicesf
GatherV2_12/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2_12/axisÍ
GatherV2_12GatherV2link_update/dense_12/Tanh_5:y:0GatherV2_12/indices:output:0GatherV2_12/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:	2
GatherV2_12	
GatherV2_13/indicesConst*
_output_shapes	
:*
dtype0*ą
value§B¤"      
                                  #   %   &   '   5   6   7   8   9          D   E   F   G   H   I                                                                         !   "                                 B   C   D   E   F   G   H   I   :   ;   <   =   >   ?   @   A         
   D   E   F   G   H   I   2   3   4   :   ;   <   =   >   ?   @   A   *   +   ,   -   :   ;   <   =   >   ?   @   A   (   )   D   E   F   G   H   I   .   /   0   1   :   ;   <   =   >   ?   @   A   *   +   ,   -   2   3   4   5   6   7   8   9   :   ;   <   =   >   ?   @   A   #   %   &   '   .   /   0   1   5   6   7   8   9         
   .   /   0   1   2   3   4   D   E   F   G   H   I   :   ;   <   =   >   ?   @   A          !   "   #   %   &   '   (   )   *   +   ,   -   .   /   0   1   5   6   7   8   9   B   C   D   E   F   G   H   I          !   "   :   ;   <   =   >   ?   @   A                      !   "   #   %   &   '   *   +   ,   -   5   6   7   8   9   :   ;   <   =   >   ?   @   A   2
GatherV2_13/indicesf
GatherV2_13/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2_13/axisÍ
GatherV2_13GatherV2link_update/dense_12/Tanh_5:y:0GatherV2_13/indices:output:0GatherV2_13/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:	2
GatherV2_13b
concat_12/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat_12/axis
	concat_12ConcatV2GatherV2_12:output:0GatherV2_13:output:0concat_12/axis:output:0*
N*
T0*
_output_shapes
:	 2
	concat_12Ö
.create_message/dense_8/MatMul_6/ReadVariableOpReadVariableOp5create_message_dense_8_matmul_readvariableop_resource*
_output_shapes

: @*
dtype020
.create_message/dense_8/MatMul_6/ReadVariableOpÂ
create_message/dense_8/MatMul_6MatMulconcat_12:output:06create_message/dense_8/MatMul_6/ReadVariableOp:value:0*
T0*
_output_shapes
:	@2!
create_message/dense_8/MatMul_6Ő
/create_message/dense_8/BiasAdd_6/ReadVariableOpReadVariableOp6create_message_dense_8_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype021
/create_message/dense_8/BiasAdd_6/ReadVariableOpÝ
 create_message/dense_8/BiasAdd_6BiasAdd)create_message/dense_8/MatMul_6:product:07create_message/dense_8/BiasAdd_6/ReadVariableOp:value:0*
T0*
_output_shapes
:	@2"
 create_message/dense_8/BiasAdd_6
create_message/dense_8/Tanh_6Tanh)create_message/dense_8/BiasAdd_6:output:0*
T0*
_output_shapes
:	@2
create_message/dense_8/Tanh_6Ö
.create_message/dense_9/MatMul_6/ReadVariableOpReadVariableOp5create_message_dense_9_matmul_readvariableop_resource*
_output_shapes

:@*
dtype020
.create_message/dense_9/MatMul_6/ReadVariableOpŃ
create_message/dense_9/MatMul_6MatMul!create_message/dense_8/Tanh_6:y:06create_message/dense_9/MatMul_6/ReadVariableOp:value:0*
T0*
_output_shapes
:	2!
create_message/dense_9/MatMul_6Ő
/create_message/dense_9/BiasAdd_6/ReadVariableOpReadVariableOp6create_message_dense_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype021
/create_message/dense_9/BiasAdd_6/ReadVariableOpÝ
 create_message/dense_9/BiasAdd_6BiasAdd)create_message/dense_9/MatMul_6:product:07create_message/dense_9/BiasAdd_6/ReadVariableOp:value:0*
T0*
_output_shapes
:	2"
 create_message/dense_9/BiasAdd_6
create_message/dense_9/Tanh_6Tanh)create_message/dense_9/BiasAdd_6:output:0*
T0*
_output_shapes
:	2
create_message/dense_9/Tanh_6¸
PartitionedCall_6PartitionedCall!create_message/dense_9/Tanh_6:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:J * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *-
f(R&
$__inference_message_aggregation_12562
PartitionedCall_6b
concat_13/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat_13/axisŞ
	concat_13ConcatV2link_update/dense_12/Tanh_5:y:0PartitionedCall_6:output:0concat_13/axis:output:0*
N*
T0*
_output_shapes

:J02
	concat_13Ń
,link_update/dense_10/MatMul_6/ReadVariableOpReadVariableOp3link_update_dense_10_matmul_readvariableop_resource*
_output_shapes
:	0*
dtype02.
,link_update/dense_10/MatMul_6/ReadVariableOpź
link_update/dense_10/MatMul_6MatMulconcat_13:output:04link_update/dense_10/MatMul_6/ReadVariableOp:value:0*
T0*
_output_shapes
:	J2
link_update/dense_10/MatMul_6Đ
-link_update/dense_10/BiasAdd_6/ReadVariableOpReadVariableOp4link_update_dense_10_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02/
-link_update/dense_10/BiasAdd_6/ReadVariableOpŐ
link_update/dense_10/BiasAdd_6BiasAdd'link_update/dense_10/MatMul_6:product:05link_update/dense_10/BiasAdd_6/ReadVariableOp:value:0*
T0*
_output_shapes
:	J2 
link_update/dense_10/BiasAdd_6
link_update/dense_10/Tanh_6Tanh'link_update/dense_10/BiasAdd_6:output:0*
T0*
_output_shapes
:	J2
link_update/dense_10/Tanh_6Ń
,link_update/dense_11/MatMul_6/ReadVariableOpReadVariableOp3link_update_dense_11_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype02.
,link_update/dense_11/MatMul_6/ReadVariableOpČ
link_update/dense_11/MatMul_6MatMullink_update/dense_10/Tanh_6:y:04link_update/dense_11/MatMul_6/ReadVariableOp:value:0*
T0*
_output_shapes

:J@2
link_update/dense_11/MatMul_6Ď
-link_update/dense_11/BiasAdd_6/ReadVariableOpReadVariableOp4link_update_dense_11_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-link_update/dense_11/BiasAdd_6/ReadVariableOpÔ
link_update/dense_11/BiasAdd_6BiasAdd'link_update/dense_11/MatMul_6:product:05link_update/dense_11/BiasAdd_6/ReadVariableOp:value:0*
T0*
_output_shapes

:J@2 
link_update/dense_11/BiasAdd_6
link_update/dense_11/Tanh_6Tanh'link_update/dense_11/BiasAdd_6:output:0*
T0*
_output_shapes

:J@2
link_update/dense_11/Tanh_6Đ
,link_update/dense_12/MatMul_6/ReadVariableOpReadVariableOp3link_update_dense_12_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02.
,link_update/dense_12/MatMul_6/ReadVariableOpČ
link_update/dense_12/MatMul_6MatMullink_update/dense_11/Tanh_6:y:04link_update/dense_12/MatMul_6/ReadVariableOp:value:0*
T0*
_output_shapes

:J2
link_update/dense_12/MatMul_6Ď
-link_update/dense_12/BiasAdd_6/ReadVariableOpReadVariableOp4link_update_dense_12_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-link_update/dense_12/BiasAdd_6/ReadVariableOpÔ
link_update/dense_12/BiasAdd_6BiasAdd'link_update/dense_12/MatMul_6:product:05link_update/dense_12/BiasAdd_6/ReadVariableOp:value:0*
T0*
_output_shapes

:J2 
link_update/dense_12/BiasAdd_6
link_update/dense_12/Tanh_6Tanh'link_update/dense_12/BiasAdd_6:output:0*
T0*
_output_shapes

:J2
link_update/dense_12/Tanh_6	
GatherV2_14/indicesConst*
_output_shapes	
:*
dtype0*ą
value§B¤"                                                      
   
   
   
   
                                                                                                                                            !   !   !   !   !   !   "   "   "   "   "   "   "   "   #   #   #   %   %   %   %   %   %   &   &   &   '   '   '   '   '   '   '   '   (   (   (   (   )   )   )   )   )   )   )   )   *   *   +   +   +   +   +   +   ,   ,   ,   ,   -   -   -   -   -   -   -   -   .   .   .   .   /   /   /   0   0   0   0   0   1   1   1   1   1   1   1   1   2   2   2   2   3   3   3   3   4   4   4   4   4   5   5   5   6   6   6   6   7   7   7   8   8   8   8   8   8   9   9   9   9   9   9   9   9   :   :   :   :   ;   ;   ;   ;   <   <   =   =   =   =   >   >   >   >   ?   ?   ?   ?   ?   @   @   A   A   A   A   A   A   B   B   B   B   C   C   C   C   C   C   C   C   D   D   D   D   E   E   E   E   F   F   F   F   G   G   G   G   H   H   H   H   H   I   I   I   I   I   I   I   I   2
GatherV2_14/indicesf
GatherV2_14/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2_14/axisÍ
GatherV2_14GatherV2link_update/dense_12/Tanh_6:y:0GatherV2_14/indices:output:0GatherV2_14/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:	2
GatherV2_14	
GatherV2_15/indicesConst*
_output_shapes	
:*
dtype0*ą
value§B¤"      
                                  #   %   &   '   5   6   7   8   9          D   E   F   G   H   I                                                                         !   "                                 B   C   D   E   F   G   H   I   :   ;   <   =   >   ?   @   A         
   D   E   F   G   H   I   2   3   4   :   ;   <   =   >   ?   @   A   *   +   ,   -   :   ;   <   =   >   ?   @   A   (   )   D   E   F   G   H   I   .   /   0   1   :   ;   <   =   >   ?   @   A   *   +   ,   -   2   3   4   5   6   7   8   9   :   ;   <   =   >   ?   @   A   #   %   &   '   .   /   0   1   5   6   7   8   9         
   .   /   0   1   2   3   4   D   E   F   G   H   I   :   ;   <   =   >   ?   @   A          !   "   #   %   &   '   (   )   *   +   ,   -   .   /   0   1   5   6   7   8   9   B   C   D   E   F   G   H   I          !   "   :   ;   <   =   >   ?   @   A                      !   "   #   %   &   '   *   +   ,   -   5   6   7   8   9   :   ;   <   =   >   ?   @   A   2
GatherV2_15/indicesf
GatherV2_15/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2_15/axisÍ
GatherV2_15GatherV2link_update/dense_12/Tanh_6:y:0GatherV2_15/indices:output:0GatherV2_15/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:	2
GatherV2_15b
concat_14/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat_14/axis
	concat_14ConcatV2GatherV2_14:output:0GatherV2_15:output:0concat_14/axis:output:0*
N*
T0*
_output_shapes
:	 2
	concat_14Ö
.create_message/dense_8/MatMul_7/ReadVariableOpReadVariableOp5create_message_dense_8_matmul_readvariableop_resource*
_output_shapes

: @*
dtype020
.create_message/dense_8/MatMul_7/ReadVariableOpÂ
create_message/dense_8/MatMul_7MatMulconcat_14:output:06create_message/dense_8/MatMul_7/ReadVariableOp:value:0*
T0*
_output_shapes
:	@2!
create_message/dense_8/MatMul_7Ő
/create_message/dense_8/BiasAdd_7/ReadVariableOpReadVariableOp6create_message_dense_8_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype021
/create_message/dense_8/BiasAdd_7/ReadVariableOpÝ
 create_message/dense_8/BiasAdd_7BiasAdd)create_message/dense_8/MatMul_7:product:07create_message/dense_8/BiasAdd_7/ReadVariableOp:value:0*
T0*
_output_shapes
:	@2"
 create_message/dense_8/BiasAdd_7
create_message/dense_8/Tanh_7Tanh)create_message/dense_8/BiasAdd_7:output:0*
T0*
_output_shapes
:	@2
create_message/dense_8/Tanh_7Ö
.create_message/dense_9/MatMul_7/ReadVariableOpReadVariableOp5create_message_dense_9_matmul_readvariableop_resource*
_output_shapes

:@*
dtype020
.create_message/dense_9/MatMul_7/ReadVariableOpŃ
create_message/dense_9/MatMul_7MatMul!create_message/dense_8/Tanh_7:y:06create_message/dense_9/MatMul_7/ReadVariableOp:value:0*
T0*
_output_shapes
:	2!
create_message/dense_9/MatMul_7Ő
/create_message/dense_9/BiasAdd_7/ReadVariableOpReadVariableOp6create_message_dense_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype021
/create_message/dense_9/BiasAdd_7/ReadVariableOpÝ
 create_message/dense_9/BiasAdd_7BiasAdd)create_message/dense_9/MatMul_7:product:07create_message/dense_9/BiasAdd_7/ReadVariableOp:value:0*
T0*
_output_shapes
:	2"
 create_message/dense_9/BiasAdd_7
create_message/dense_9/Tanh_7Tanh)create_message/dense_9/BiasAdd_7:output:0*
T0*
_output_shapes
:	2
create_message/dense_9/Tanh_7¸
PartitionedCall_7PartitionedCall!create_message/dense_9/Tanh_7:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:J * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *-
f(R&
$__inference_message_aggregation_12562
PartitionedCall_7b
concat_15/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat_15/axisŞ
	concat_15ConcatV2link_update/dense_12/Tanh_6:y:0PartitionedCall_7:output:0concat_15/axis:output:0*
N*
T0*
_output_shapes

:J02
	concat_15Ń
,link_update/dense_10/MatMul_7/ReadVariableOpReadVariableOp3link_update_dense_10_matmul_readvariableop_resource*
_output_shapes
:	0*
dtype02.
,link_update/dense_10/MatMul_7/ReadVariableOpź
link_update/dense_10/MatMul_7MatMulconcat_15:output:04link_update/dense_10/MatMul_7/ReadVariableOp:value:0*
T0*
_output_shapes
:	J2
link_update/dense_10/MatMul_7Đ
-link_update/dense_10/BiasAdd_7/ReadVariableOpReadVariableOp4link_update_dense_10_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02/
-link_update/dense_10/BiasAdd_7/ReadVariableOpŐ
link_update/dense_10/BiasAdd_7BiasAdd'link_update/dense_10/MatMul_7:product:05link_update/dense_10/BiasAdd_7/ReadVariableOp:value:0*
T0*
_output_shapes
:	J2 
link_update/dense_10/BiasAdd_7
link_update/dense_10/Tanh_7Tanh'link_update/dense_10/BiasAdd_7:output:0*
T0*
_output_shapes
:	J2
link_update/dense_10/Tanh_7Ń
,link_update/dense_11/MatMul_7/ReadVariableOpReadVariableOp3link_update_dense_11_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype02.
,link_update/dense_11/MatMul_7/ReadVariableOpČ
link_update/dense_11/MatMul_7MatMullink_update/dense_10/Tanh_7:y:04link_update/dense_11/MatMul_7/ReadVariableOp:value:0*
T0*
_output_shapes

:J@2
link_update/dense_11/MatMul_7Ď
-link_update/dense_11/BiasAdd_7/ReadVariableOpReadVariableOp4link_update_dense_11_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-link_update/dense_11/BiasAdd_7/ReadVariableOpÔ
link_update/dense_11/BiasAdd_7BiasAdd'link_update/dense_11/MatMul_7:product:05link_update/dense_11/BiasAdd_7/ReadVariableOp:value:0*
T0*
_output_shapes

:J@2 
link_update/dense_11/BiasAdd_7
link_update/dense_11/Tanh_7Tanh'link_update/dense_11/BiasAdd_7:output:0*
T0*
_output_shapes

:J@2
link_update/dense_11/Tanh_7Đ
,link_update/dense_12/MatMul_7/ReadVariableOpReadVariableOp3link_update_dense_12_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02.
,link_update/dense_12/MatMul_7/ReadVariableOpČ
link_update/dense_12/MatMul_7MatMullink_update/dense_11/Tanh_7:y:04link_update/dense_12/MatMul_7/ReadVariableOp:value:0*
T0*
_output_shapes

:J2
link_update/dense_12/MatMul_7Ď
-link_update/dense_12/BiasAdd_7/ReadVariableOpReadVariableOp4link_update_dense_12_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-link_update/dense_12/BiasAdd_7/ReadVariableOpÔ
link_update/dense_12/BiasAdd_7BiasAdd'link_update/dense_12/MatMul_7:product:05link_update/dense_12/BiasAdd_7/ReadVariableOp:value:0*
T0*
_output_shapes

:J2 
link_update/dense_12/BiasAdd_7
link_update/dense_12/Tanh_7Tanh'link_update/dense_12/BiasAdd_7:output:0*
T0*
_output_shapes

:J2
link_update/dense_12/Tanh_7j
IdentityIdentitylink_update/dense_12/Tanh_7:y:0*
T0*
_output_shapes

:J2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:˙˙˙˙˙˙˙˙˙:::::::::::J F
#
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_nameinput
É%
Đ
A__inference_critic_layer_call_and_return_conditional_losses_54525
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_83
/readout_dense_13_matmul_readvariableop_resource4
0readout_dense_13_biasadd_readvariableop_resource3
/readout_dense_14_matmul_readvariableop_resource4
0readout_dense_14_biasadd_readvariableop_resource3
/readout_dense_15_matmul_readvariableop_resource4
0readout_dense_15_biasadd_readvariableop_resource
identity˘StatefulPartitionedCallą
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:J*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8 **
f%R#
!__inference_message_passing_424792
StatefulPartitionedCallś
PartitionedCallPartitionedCall StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *0
f+R)
'__inference_generate_readout_input_15772
PartitionedCallÁ
&readout/dense_13/MatMul/ReadVariableOpReadVariableOp/readout_dense_13_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype02(
&readout/dense_13/MatMul/ReadVariableOp°
readout/dense_13/MatMulMatMulPartitionedCall:output:0.readout/dense_13/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
readout/dense_13/MatMulŔ
'readout/dense_13/BiasAdd/ReadVariableOpReadVariableOp0readout_dense_13_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02)
'readout/dense_13/BiasAdd/ReadVariableOp˝
readout/dense_13/BiasAddBiasAdd!readout/dense_13/MatMul:product:0/readout/dense_13/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
readout/dense_13/BiasAdd
readout/dense_13/TanhTanh!readout/dense_13/BiasAdd:output:0*
T0*
_output_shapes
:	2
readout/dense_13/Tanh
readout/dropout_2/IdentityIdentityreadout/dense_13/Tanh:y:0*
T0*
_output_shapes
:	2
readout/dropout_2/IdentityÁ
&readout/dense_14/MatMul/ReadVariableOpReadVariableOp/readout_dense_14_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype02(
&readout/dense_14/MatMul/ReadVariableOpş
readout/dense_14/MatMulMatMul#readout/dropout_2/Identity:output:0.readout/dense_14/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:@2
readout/dense_14/MatMulż
'readout/dense_14/BiasAdd/ReadVariableOpReadVariableOp0readout_dense_14_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02)
'readout/dense_14/BiasAdd/ReadVariableOpź
readout/dense_14/BiasAddBiasAdd!readout/dense_14/MatMul:product:0/readout/dense_14/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:@2
readout/dense_14/BiasAdd
readout/dense_14/TanhTanh!readout/dense_14/BiasAdd:output:0*
T0*
_output_shapes

:@2
readout/dense_14/Tanh
readout/dropout_3/IdentityIdentityreadout/dense_14/Tanh:y:0*
T0*
_output_shapes

:@2
readout/dropout_3/IdentityŔ
&readout/dense_15/MatMul/ReadVariableOpReadVariableOp/readout_dense_15_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02(
&readout/dense_15/MatMul/ReadVariableOpş
readout/dense_15/MatMulMatMul#readout/dropout_3/Identity:output:0.readout/dense_15/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
readout/dense_15/MatMulż
'readout/dense_15/BiasAdd/ReadVariableOpReadVariableOp0readout_dense_15_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'readout/dense_15/BiasAdd/ReadVariableOpź
readout/dense_15/BiasAddBiasAdd!readout/dense_15/MatMul:product:0/readout/dense_15/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:2
readout/dense_15/BiasAddq
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙2
Reshape/shape}
ReshapeReshape!readout/dense_15/BiasAdd:output:0Reshape/shape:output:0*
T0*
_output_shapes
:2	
Reshapeq
IdentityIdentityReshape:output:0^StatefulPartitionedCall*
T0*
_output_shapes
:2

Identity"
identityIdentity:output:0*b
_input_shapesQ
O:˙˙˙˙˙˙˙˙˙::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:L H
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
!
_user_specified_name	input_1
š
B
$__inference_message_aggregation_1256
messages
identity§	
UnsortedSegmentMax/segment_idsConst*
_output_shapes	
:*
dtype0*ą
value§B¤"      
                                  #   %   &   '   5   6   7   8   9          D   E   F   G   H   I                                                                         !   "                                 B   C   D   E   F   G   H   I   :   ;   <   =   >   ?   @   A         
   D   E   F   G   H   I   2   3   4   :   ;   <   =   >   ?   @   A   *   +   ,   -   :   ;   <   =   >   ?   @   A   (   )   D   E   F   G   H   I   .   /   0   1   :   ;   <   =   >   ?   @   A   *   +   ,   -   2   3   4   5   6   7   8   9   :   ;   <   =   >   ?   @   A   #   %   &   '   .   /   0   1   5   6   7   8   9         
   .   /   0   1   2   3   4   D   E   F   G   H   I   :   ;   <   =   >   ?   @   A          !   "   #   %   &   '   (   )   *   +   ,   -   .   /   0   1   5   6   7   8   9   B   C   D   E   F   G   H   I          !   "   :   ;   <   =   >   ?   @   A                      !   "   #   %   &   '   *   +   ,   -   5   6   7   8   9   :   ;   <   =   >   ?   @   A   2 
UnsortedSegmentMax/segment_ids
UnsortedSegmentMax/num_segmentsConst*
_output_shapes
: *
dtype0*
value	B :J2!
UnsortedSegmentMax/num_segmentsÔ
UnsortedSegmentMaxUnsortedSegmentMaxmessages'UnsortedSegmentMax/segment_ids:output:0(UnsortedSegmentMax/num_segments:output:0*
T0*
Tindices0*
_output_shapes

:J2
UnsortedSegmentMax§	
UnsortedSegmentMin/segment_idsConst*
_output_shapes	
:*
dtype0*ą
value§B¤"      
                                  #   %   &   '   5   6   7   8   9          D   E   F   G   H   I                                                                         !   "                                 B   C   D   E   F   G   H   I   :   ;   <   =   >   ?   @   A         
   D   E   F   G   H   I   2   3   4   :   ;   <   =   >   ?   @   A   *   +   ,   -   :   ;   <   =   >   ?   @   A   (   )   D   E   F   G   H   I   .   /   0   1   :   ;   <   =   >   ?   @   A   *   +   ,   -   2   3   4   5   6   7   8   9   :   ;   <   =   >   ?   @   A   #   %   &   '   .   /   0   1   5   6   7   8   9         
   .   /   0   1   2   3   4   D   E   F   G   H   I   :   ;   <   =   >   ?   @   A          !   "   #   %   &   '   (   )   *   +   ,   -   .   /   0   1   5   6   7   8   9   B   C   D   E   F   G   H   I          !   "   :   ;   <   =   >   ?   @   A                      !   "   #   %   &   '   *   +   ,   -   5   6   7   8   9   :   ;   <   =   >   ?   @   A   2 
UnsortedSegmentMin/segment_ids
UnsortedSegmentMin/num_segmentsConst*
_output_shapes
: *
dtype0*
value	B :J2!
UnsortedSegmentMin/num_segmentsÔ
UnsortedSegmentMinUnsortedSegmentMinmessages'UnsortedSegmentMin/segment_ids:output:0(UnsortedSegmentMin/num_segments:output:0*
T0*
Tindices0*
_output_shapes

:J2
UnsortedSegmentMin\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis
concatConcatV2UnsortedSegmentMax:output:0UnsortedSegmentMin:output:0concat/axis:output:0*
N*
T0*
_output_shapes

:J 2
concatZ
IdentityIdentityconcat:output:0*
T0*
_output_shapes

:J 2

Identity"
identityIdentity:output:0*
_input_shapes
:	:I E

_output_shapes
:	
"
_user_specified_name
messages
Ď
ň
B__inference_readout_layer_call_and_return_conditional_losses_55099

inputs+
'dense_13_matmul_readvariableop_resource,
(dense_13_biasadd_readvariableop_resource+
'dense_14_matmul_readvariableop_resource,
(dense_14_biasadd_readvariableop_resource+
'dense_15_matmul_readvariableop_resource,
(dense_15_biasadd_readvariableop_resource
identityŠ
dense_13/MatMul/ReadVariableOpReadVariableOp'dense_13_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype02 
dense_13/MatMul/ReadVariableOp
dense_13/MatMulMatMulinputs&dense_13/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
dense_13/MatMul¨
dense_13/BiasAdd/ReadVariableOpReadVariableOp(dense_13_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02!
dense_13/BiasAdd/ReadVariableOpŚ
dense_13/BiasAddBiasAdddense_13/MatMul:product:0'dense_13/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
dense_13/BiasAddt
dense_13/TanhTanhdense_13/BiasAdd:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
dense_13/Tanhz
dropout_2/IdentityIdentitydense_13/Tanh:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
dropout_2/IdentityŠ
dense_14/MatMul/ReadVariableOpReadVariableOp'dense_14_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype02 
dense_14/MatMul/ReadVariableOpŁ
dense_14/MatMulMatMuldropout_2/Identity:output:0&dense_14/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@2
dense_14/MatMul§
dense_14/BiasAdd/ReadVariableOpReadVariableOp(dense_14_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
dense_14/BiasAdd/ReadVariableOpĽ
dense_14/BiasAddBiasAdddense_14/MatMul:product:0'dense_14/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@2
dense_14/BiasAdds
dense_14/TanhTanhdense_14/BiasAdd:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@2
dense_14/Tanhy
dropout_3/IdentityIdentitydense_14/Tanh:y:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@2
dropout_3/Identity¨
dense_15/MatMul/ReadVariableOpReadVariableOp'dense_15_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02 
dense_15/MatMul/ReadVariableOpŁ
dense_15/MatMulMatMuldropout_3/Identity:output:0&dense_15/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2
dense_15/MatMul§
dense_15/BiasAdd/ReadVariableOpReadVariableOp(dense_15_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_15/BiasAdd/ReadVariableOpĽ
dense_15/BiasAddBiasAdddense_15/MatMul:product:0'dense_15/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2
dense_15/BiasAddm
IdentityIdentitydense_15/BiasAdd:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:˙˙˙˙˙˙˙˙˙@:::::::O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙@
 
_user_specified_nameinputs
Ď
Î
F__inference_link_update_layer_call_and_return_conditional_losses_53738
dense_10_input
dense_10_53678
dense_10_53680
dense_11_53705
dense_11_53707
dense_12_53732
dense_12_53734
identity˘ dense_10/StatefulPartitionedCall˘ dense_11/StatefulPartitionedCall˘ dense_12/StatefulPartitionedCall
 dense_10/StatefulPartitionedCallStatefulPartitionedCalldense_10_inputdense_10_53678dense_10_53680*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_10_layer_call_and_return_conditional_losses_536672"
 dense_10/StatefulPartitionedCall´
 dense_11/StatefulPartitionedCallStatefulPartitionedCall)dense_10/StatefulPartitionedCall:output:0dense_11_53705dense_11_53707*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_11_layer_call_and_return_conditional_losses_536942"
 dense_11/StatefulPartitionedCall´
 dense_12/StatefulPartitionedCallStatefulPartitionedCall)dense_11/StatefulPartitionedCall:output:0dense_12_53732dense_12_53734*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_12_layer_call_and_return_conditional_losses_537212"
 dense_12/StatefulPartitionedCallć
IdentityIdentity)dense_12/StatefulPartitionedCall:output:0!^dense_10/StatefulPartitionedCall!^dense_11/StatefulPartitionedCall!^dense_12/StatefulPartitionedCall*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:˙˙˙˙˙˙˙˙˙0::::::2D
 dense_10/StatefulPartitionedCall dense_10/StatefulPartitionedCall2D
 dense_11/StatefulPartitionedCall dense_11/StatefulPartitionedCall2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall:W S
'
_output_shapes
:˙˙˙˙˙˙˙˙˙0
(
_user_specified_namedense_10_input
Ř
|
'__inference_dense_9_layer_call_fn_55173

inputs
unknown
	unknown_0
identity˘StatefulPartitionedCallň
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_9_layer_call_and_return_conditional_losses_535662
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*.
_input_shapes
:˙˙˙˙˙˙˙˙˙@::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙@
 
_user_specified_nameinputs
Ç
b
D__inference_dropout_3_layer_call_and_return_conditional_losses_55317

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@2

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:˙˙˙˙˙˙˙˙˙@:O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙@
 
_user_specified_nameinputs
ń
Ŕ
'__inference_readout_layer_call_fn_54073
dense_13_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity˘StatefulPartitionedCallŽ
StatefulPartitionedCallStatefulPartitionedCalldense_13_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_readout_layer_call_and_return_conditional_losses_540582
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:˙˙˙˙˙˙˙˙˙@::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
'
_output_shapes
:˙˙˙˙˙˙˙˙˙@
(
_user_specified_namedense_13_input
Ě
Ť
C__inference_dense_15_layer_call_and_return_conditional_losses_55337

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2	
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*.
_input_shapes
:˙˙˙˙˙˙˙˙˙@:::O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙@
 
_user_specified_nameinputs
ü	
Ń
&__inference_critic_layer_call_fn_54787	
input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14
identity˘StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_critic_layer_call_and_return_conditional_losses_543002
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes
:2

Identity"
identityIdentity:output:0*b
_input_shapesQ
O:˙˙˙˙˙˙˙˙˙::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:J F
#
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_nameinput
Ç
b
D__inference_dropout_3_layer_call_and_return_conditional_losses_53935

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@2

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:˙˙˙˙˙˙˙˙˙@:O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙@
 
_user_specified_nameinputs

c
D__inference_dropout_2_layer_call_and_return_conditional_losses_53873

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shapeľ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2
dropout/GreaterEqual/yż
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*'
_input_shapes
:˙˙˙˙˙˙˙˙˙:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs


I__inference_create_message_layer_call_and_return_conditional_losses_54805

inputs*
&dense_8_matmul_readvariableop_resource+
'dense_8_biasadd_readvariableop_resource*
&dense_9_matmul_readvariableop_resource+
'dense_9_biasadd_readvariableop_resource
identityĽ
dense_8/MatMul/ReadVariableOpReadVariableOp&dense_8_matmul_readvariableop_resource*
_output_shapes

: @*
dtype02
dense_8/MatMul/ReadVariableOp
dense_8/MatMulMatMulinputs%dense_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@2
dense_8/MatMul¤
dense_8/BiasAdd/ReadVariableOpReadVariableOp'dense_8_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02 
dense_8/BiasAdd/ReadVariableOpĄ
dense_8/BiasAddBiasAdddense_8/MatMul:product:0&dense_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@2
dense_8/BiasAddp
dense_8/TanhTanhdense_8/BiasAdd:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@2
dense_8/TanhĽ
dense_9/MatMul/ReadVariableOpReadVariableOp&dense_9_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02
dense_9/MatMul/ReadVariableOp
dense_9/MatMulMatMuldense_8/Tanh:y:0%dense_9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2
dense_9/MatMul¤
dense_9/BiasAdd/ReadVariableOpReadVariableOp'dense_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_9/BiasAdd/ReadVariableOpĄ
dense_9/BiasAddBiasAdddense_9/MatMul:product:0&dense_9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2
dense_9/BiasAddp
dense_9/TanhTanhdense_9/BiasAdd:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2
dense_9/Tanhd
IdentityIdentitydense_9/Tanh:y:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:˙˙˙˙˙˙˙˙˙ :::::O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
 
_user_specified_nameinputs
ú)
ň
B__inference_readout_layer_call_and_return_conditional_losses_54973

inputs+
'dense_13_matmul_readvariableop_resource,
(dense_13_biasadd_readvariableop_resource+
'dense_14_matmul_readvariableop_resource,
(dense_14_biasadd_readvariableop_resource+
'dense_15_matmul_readvariableop_resource,
(dense_15_biasadd_readvariableop_resource
identityŠ
dense_13/MatMul/ReadVariableOpReadVariableOp'dense_13_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype02 
dense_13/MatMul/ReadVariableOp
dense_13/MatMulMatMulinputs&dense_13/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
dense_13/MatMul¨
dense_13/BiasAdd/ReadVariableOpReadVariableOp(dense_13_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02!
dense_13/BiasAdd/ReadVariableOp
dense_13/BiasAddBiasAdddense_13/MatMul:product:0'dense_13/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
dense_13/BiasAddk
dense_13/TanhTanhdense_13/BiasAdd:output:0*
T0*
_output_shapes
:	2
dense_13/Tanhw
dropout_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?2
dropout_2/dropout/Const
dropout_2/dropout/MulMuldense_13/Tanh:y:0 dropout_2/dropout/Const:output:0*
T0*
_output_shapes
:	2
dropout_2/dropout/Mul
dropout_2/dropout/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2
dropout_2/dropout/ShapeĘ
.dropout_2/dropout/random_uniform/RandomUniformRandomUniform dropout_2/dropout/Shape:output:0*
T0*
_output_shapes
:	*
dtype020
.dropout_2/dropout/random_uniform/RandomUniform
 dropout_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2"
 dropout_2/dropout/GreaterEqual/yŢ
dropout_2/dropout/GreaterEqualGreaterEqual7dropout_2/dropout/random_uniform/RandomUniform:output:0)dropout_2/dropout/GreaterEqual/y:output:0*
T0*
_output_shapes
:	2 
dropout_2/dropout/GreaterEqual
dropout_2/dropout/CastCast"dropout_2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*
_output_shapes
:	2
dropout_2/dropout/Cast
dropout_2/dropout/Mul_1Muldropout_2/dropout/Mul:z:0dropout_2/dropout/Cast:y:0*
T0*
_output_shapes
:	2
dropout_2/dropout/Mul_1Š
dense_14/MatMul/ReadVariableOpReadVariableOp'dense_14_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype02 
dense_14/MatMul/ReadVariableOp
dense_14/MatMulMatMuldropout_2/dropout/Mul_1:z:0&dense_14/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:@2
dense_14/MatMul§
dense_14/BiasAdd/ReadVariableOpReadVariableOp(dense_14_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
dense_14/BiasAdd/ReadVariableOp
dense_14/BiasAddBiasAdddense_14/MatMul:product:0'dense_14/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:@2
dense_14/BiasAddj
dense_14/TanhTanhdense_14/BiasAdd:output:0*
T0*
_output_shapes

:@2
dense_14/Tanhw
dropout_3/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?2
dropout_3/dropout/Const
dropout_3/dropout/MulMuldense_14/Tanh:y:0 dropout_3/dropout/Const:output:0*
T0*
_output_shapes

:@2
dropout_3/dropout/Mul
dropout_3/dropout/ShapeConst*
_output_shapes
:*
dtype0*
valueB"   @   2
dropout_3/dropout/ShapeÉ
.dropout_3/dropout/random_uniform/RandomUniformRandomUniform dropout_3/dropout/Shape:output:0*
T0*
_output_shapes

:@*
dtype020
.dropout_3/dropout/random_uniform/RandomUniform
 dropout_3/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2"
 dropout_3/dropout/GreaterEqual/yÝ
dropout_3/dropout/GreaterEqualGreaterEqual7dropout_3/dropout/random_uniform/RandomUniform:output:0)dropout_3/dropout/GreaterEqual/y:output:0*
T0*
_output_shapes

:@2 
dropout_3/dropout/GreaterEqual
dropout_3/dropout/CastCast"dropout_3/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*
_output_shapes

:@2
dropout_3/dropout/Cast
dropout_3/dropout/Mul_1Muldropout_3/dropout/Mul:z:0dropout_3/dropout/Cast:y:0*
T0*
_output_shapes

:@2
dropout_3/dropout/Mul_1¨
dense_15/MatMul/ReadVariableOpReadVariableOp'dense_15_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02 
dense_15/MatMul/ReadVariableOp
dense_15/MatMulMatMuldropout_3/dropout/Mul_1:z:0&dense_15/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
dense_15/MatMul§
dense_15/BiasAdd/ReadVariableOpReadVariableOp(dense_15_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_15/BiasAdd/ReadVariableOp
dense_15/BiasAddBiasAdddense_15/MatMul:product:0'dense_15/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:2
dense_15/BiasAddd
IdentityIdentitydense_15/BiasAdd:output:0*
T0*
_output_shapes

:2

Identity"
identityIdentity:output:0*5
_input_shapes$
":@:::::::F B

_output_shapes

:@
 
_user_specified_nameinputs
Ą
Ť
C__inference_dense_11_layer_call_and_return_conditional_losses_55204

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@2	
BiasAddX
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@2
Tanh\
IdentityIdentityTanh:y:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@2

Identity"
identityIdentity:output:0*/
_input_shapes
:˙˙˙˙˙˙˙˙˙:::P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
¤
Ť
C__inference_dense_10_layer_call_and_return_conditional_losses_55184

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	0*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2	
BiasAddY
TanhTanhBiasAdd:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
Tanh]
IdentityIdentityTanh:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*.
_input_shapes
:˙˙˙˙˙˙˙˙˙0:::O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙0
 
_user_specified_nameinputs
§
Ą
.__inference_create_message_layer_call_fn_54849

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity˘StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_create_message_layer_call_and_return_conditional_losses_536412
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:˙˙˙˙˙˙˙˙˙ ::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
 
_user_specified_nameinputs
Ü


 __inference__wrapped_model_53524
input_1
critic_53490
critic_53492
critic_53494
critic_53496
critic_53498
critic_53500
critic_53502
critic_53504
critic_53506
critic_53508
critic_53510
critic_53512
critic_53514
critic_53516
critic_53518
critic_53520
identity˘critic/StatefulPartitionedCall°
critic/StatefulPartitionedCallStatefulPartitionedCallinput_1critic_53490critic_53492critic_53494critic_53496critic_53498critic_53500critic_53502critic_53504critic_53506critic_53508critic_53510critic_53512critic_53514critic_53516critic_53518critic_53520*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *
fR
__inference_call_425272 
critic/StatefulPartitionedCall
IdentityIdentity'critic/StatefulPartitionedCall:output:0^critic/StatefulPartitionedCall*
T0*
_output_shapes
:2

Identity"
identityIdentity:output:0*b
_input_shapesQ
O:˙˙˙˙˙˙˙˙˙::::::::::::::::2@
critic/StatefulPartitionedCallcritic/StatefulPartitionedCall:L H
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
!
_user_specified_name	input_1

c
D__inference_dropout_3_layer_call_and_return_conditional_losses_55312

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape´
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2
dropout/GreaterEqual/yž
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@2

Identity"
identityIdentity:output:0*&
_input_shapes
:˙˙˙˙˙˙˙˙˙@:O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙@
 
_user_specified_nameinputs

c
D__inference_dropout_3_layer_call_and_return_conditional_losses_53930

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape´
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2
dropout/GreaterEqual/yž
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@2

Identity"
identityIdentity:output:0*&
_input_shapes
:˙˙˙˙˙˙˙˙˙@:O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙@
 
_user_specified_nameinputs"¸L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*
serving_default
7
input_1,
serving_default_input_1:0˙˙˙˙˙˙˙˙˙/
output_1#
StatefulPartitionedCall:0tensorflow/serving/predict:˛
ć
incoming_links
outcoming_links
create_message
link_update
readout
regularization_losses
trainable_variables
	variables
		keras_api


signatures
Š__call__
Ş_default_save_signature
+Ť&call_and_return_all_conditional_losses
	Źcall
­generate_readout_input
Žmessage_aggregation
Żmessage_passing"ô
_tf_keras_modelÚ{"class_name": "Critic", "name": "critic", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"layer was saved without config": true}, "is_graph_network": false, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Critic"}}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
ű
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
regularization_losses
trainable_variables
	variables
	keras_api
°__call__
+ą&call_and_return_all_conditional_losses"
_tf_keras_sequentialý{"class_name": "Sequential", "name": "create_message", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "create_message", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 32]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_8_input"}}, {"class_name": "Dense", "config": {"name": "dense_8", "trainable": true, "dtype": "float32", "units": 64, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.4142135623730951, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_9", "trainable": true, "dtype": "float32", "units": 16, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.4142135623730951, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "create_message", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 32]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_8_input"}}, {"class_name": "Dense", "config": {"name": "dense_8", "trainable": true, "dtype": "float32", "units": 64, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.4142135623730951, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_9", "trainable": true, "dtype": "float32", "units": 16, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.4142135623730951, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}}
Ľ
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
regularization_losses
trainable_variables
	variables
	keras_api
˛__call__
+ł&call_and_return_all_conditional_losses"
_tf_keras_sequential{"class_name": "Sequential", "name": "link_update", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "link_update", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 48]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_10_input"}}, {"class_name": "Dense", "config": {"name": "dense_10", "trainable": true, "dtype": "float32", "units": 128, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.4142135623730951, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_11", "trainable": true, "dtype": "float32", "units": 64, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.4142135623730951, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_12", "trainable": true, "dtype": "float32", "units": 16, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.4142135623730951, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 48}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 48]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "link_update", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 48]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_10_input"}}, {"class_name": "Dense", "config": {"name": "dense_10", "trainable": true, "dtype": "float32", "units": 128, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.4142135623730951, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_11", "trainable": true, "dtype": "float32", "units": 64, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.4142135623730951, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_12", "trainable": true, "dtype": "float32", "units": 16, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.4142135623730951, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}}
ă#
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
regularization_losses
trainable_variables
	variables
 	keras_api
´__call__
+ľ&call_and_return_all_conditional_losses"Ă!
_tf_keras_sequential¤!{"class_name": "Sequential", "name": "readout", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "readout", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 64]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_13_input"}}, {"class_name": "Dense", "config": {"name": "dense_13", "trainable": true, "dtype": "float32", "units": 128, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.4142135623730951, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_2", "trainable": true, "dtype": "float32", "rate": 0.15, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_14", "trainable": true, "dtype": "float32", "units": 64, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.4142135623730951, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_3", "trainable": true, "dtype": "float32", "rate": 0.15, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_15", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "Orthogonal", "config": {"gain": 1, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "readout", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 64]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_13_input"}}, {"class_name": "Dense", "config": {"name": "dense_13", "trainable": true, "dtype": "float32", "units": 128, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.4142135623730951, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_2", "trainable": true, "dtype": "float32", "rate": 0.15, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_14", "trainable": true, "dtype": "float32", "units": 64, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.4142135623730951, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_3", "trainable": true, "dtype": "float32", "rate": 0.15, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_15", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "Orthogonal", "config": {"gain": 1, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}}
 "
trackable_list_wrapper

!0
"1
#2
$3
%4
&5
'6
(7
)8
*9
+10
,11
-12
.13
/14
015"
trackable_list_wrapper

!0
"1
#2
$3
%4
&5
'6
(7
)8
*9
+10
,11
-12
.13
/14
015"
trackable_list_wrapper
Î
1non_trainable_variables

2layers
regularization_losses
trainable_variables
	variables
3layer_metrics
4layer_regularization_losses
5metrics
Š__call__
Ş_default_save_signature
+Ť&call_and_return_all_conditional_losses
'Ť"call_and_return_conditional_losses"
_generic_user_object
-
śserving_default"
signature_map

6_inbound_nodes

!kernel
"bias
7regularization_losses
8trainable_variables
9	variables
:	keras_api
ˇ__call__
+¸&call_and_return_all_conditional_losses"ä
_tf_keras_layerĘ{"class_name": "Dense", "name": "dense_8", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_8", "trainable": true, "dtype": "float32", "units": 64, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.4142135623730951, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32]}}

;_inbound_nodes

#kernel
$bias
<regularization_losses
=trainable_variables
>	variables
?	keras_api
š__call__
+ş&call_and_return_all_conditional_losses"ä
_tf_keras_layerĘ{"class_name": "Dense", "name": "dense_9", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_9", "trainable": true, "dtype": "float32", "units": 16, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.4142135623730951, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64]}}
 "
trackable_list_wrapper
<
!0
"1
#2
$3"
trackable_list_wrapper
<
!0
"1
#2
$3"
trackable_list_wrapper
°
@non_trainable_variables

Alayers
regularization_losses
trainable_variables
	variables
Blayer_metrics
Clayer_regularization_losses
Dmetrics
°__call__
+ą&call_and_return_all_conditional_losses
'ą"call_and_return_conditional_losses"
_generic_user_object
˘
E_inbound_nodes

%kernel
&bias
Fregularization_losses
Gtrainable_variables
H	variables
I	keras_api
ť__call__
+ź&call_and_return_all_conditional_losses"ç
_tf_keras_layerÍ{"class_name": "Dense", "name": "dense_10", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_10", "trainable": true, "dtype": "float32", "units": 128, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.4142135623730951, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 48}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 48]}}
Ł
J_inbound_nodes

'kernel
(bias
Kregularization_losses
Ltrainable_variables
M	variables
N	keras_api
˝__call__
+ž&call_and_return_all_conditional_losses"č
_tf_keras_layerÎ{"class_name": "Dense", "name": "dense_11", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_11", "trainable": true, "dtype": "float32", "units": 64, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.4142135623730951, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128]}}
Ą
O_inbound_nodes

)kernel
*bias
Pregularization_losses
Qtrainable_variables
R	variables
S	keras_api
ż__call__
+Ŕ&call_and_return_all_conditional_losses"ć
_tf_keras_layerĚ{"class_name": "Dense", "name": "dense_12", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_12", "trainable": true, "dtype": "float32", "units": 16, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.4142135623730951, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64]}}
 "
trackable_list_wrapper
J
%0
&1
'2
(3
)4
*5"
trackable_list_wrapper
J
%0
&1
'2
(3
)4
*5"
trackable_list_wrapper
°
Tnon_trainable_variables

Ulayers
regularization_losses
trainable_variables
	variables
Vlayer_metrics
Wlayer_regularization_losses
Xmetrics
˛__call__
+ł&call_and_return_all_conditional_losses
'ł"call_and_return_conditional_losses"
_generic_user_object
˘
Y_inbound_nodes

+kernel
,bias
Zregularization_losses
[trainable_variables
\	variables
]	keras_api
Á__call__
+Â&call_and_return_all_conditional_losses"ç
_tf_keras_layerÍ{"class_name": "Dense", "name": "dense_13", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_13", "trainable": true, "dtype": "float32", "units": 128, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.4142135623730951, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64]}}
ü
^_inbound_nodes
_regularization_losses
`trainable_variables
a	variables
b	keras_api
Ă__call__
+Ä&call_and_return_all_conditional_losses"×
_tf_keras_layer˝{"class_name": "Dropout", "name": "dropout_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_2", "trainable": true, "dtype": "float32", "rate": 0.15, "noise_shape": null, "seed": null}}
Ł
c_inbound_nodes

-kernel
.bias
dregularization_losses
etrainable_variables
f	variables
g	keras_api
Ĺ__call__
+Ć&call_and_return_all_conditional_losses"č
_tf_keras_layerÎ{"class_name": "Dense", "name": "dense_14", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_14", "trainable": true, "dtype": "float32", "units": 64, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.4142135623730951, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128]}}
ü
h_inbound_nodes
iregularization_losses
jtrainable_variables
k	variables
l	keras_api
Ç__call__
+Č&call_and_return_all_conditional_losses"×
_tf_keras_layer˝{"class_name": "Dropout", "name": "dropout_3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_3", "trainable": true, "dtype": "float32", "rate": 0.15, "noise_shape": null, "seed": null}}

m_inbound_nodes

/kernel
0bias
nregularization_losses
otrainable_variables
p	variables
q	keras_api
É__call__
+Ę&call_and_return_all_conditional_losses"Ö
_tf_keras_layerź{"class_name": "Dense", "name": "dense_15", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_15", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "Orthogonal", "config": {"gain": 1, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64]}}
 "
trackable_list_wrapper
J
+0
,1
-2
.3
/4
05"
trackable_list_wrapper
J
+0
,1
-2
.3
/4
05"
trackable_list_wrapper
°
rnon_trainable_variables

slayers
regularization_losses
trainable_variables
	variables
tlayer_metrics
ulayer_regularization_losses
vmetrics
´__call__
+ľ&call_and_return_all_conditional_losses
'ľ"call_and_return_conditional_losses"
_generic_user_object
 : @2dense_8/kernel
:@2dense_8/bias
 :@2dense_9/kernel
:2dense_9/bias
": 	02dense_10/kernel
:2dense_10/bias
": 	@2dense_11/kernel
:@2dense_11/bias
!:@2dense_12/kernel
:2dense_12/bias
": 	@2dense_13/kernel
:2dense_13/bias
": 	@2dense_14/kernel
:@2dense_14/bias
!:@2dense_15/kernel
:2dense_15/bias
 "
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
!0
"1"
trackable_list_wrapper
.
!0
"1"
trackable_list_wrapper
°
wnon_trainable_variables

xlayers
7regularization_losses
ylayer_metrics
8trainable_variables
9	variables
zlayer_regularization_losses
{metrics
ˇ__call__
+¸&call_and_return_all_conditional_losses
'¸"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
#0
$1"
trackable_list_wrapper
.
#0
$1"
trackable_list_wrapper
ą
|non_trainable_variables

}layers
<regularization_losses
~layer_metrics
=trainable_variables
>	variables
layer_regularization_losses
metrics
š__call__
+ş&call_and_return_all_conditional_losses
'ş"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
%0
&1"
trackable_list_wrapper
.
%0
&1"
trackable_list_wrapper
ľ
non_trainable_variables
layers
Fregularization_losses
layer_metrics
Gtrainable_variables
H	variables
 layer_regularization_losses
metrics
ť__call__
+ź&call_and_return_all_conditional_losses
'ź"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
'0
(1"
trackable_list_wrapper
.
'0
(1"
trackable_list_wrapper
ľ
non_trainable_variables
layers
Kregularization_losses
layer_metrics
Ltrainable_variables
M	variables
 layer_regularization_losses
metrics
˝__call__
+ž&call_and_return_all_conditional_losses
'ž"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
)0
*1"
trackable_list_wrapper
.
)0
*1"
trackable_list_wrapper
ľ
non_trainable_variables
layers
Pregularization_losses
layer_metrics
Qtrainable_variables
R	variables
 layer_regularization_losses
metrics
ż__call__
+Ŕ&call_and_return_all_conditional_losses
'Ŕ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
+0
,1"
trackable_list_wrapper
.
+0
,1"
trackable_list_wrapper
ľ
non_trainable_variables
layers
Zregularization_losses
layer_metrics
[trainable_variables
\	variables
 layer_regularization_losses
metrics
Á__call__
+Â&call_and_return_all_conditional_losses
'Â"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
ľ
non_trainable_variables
layers
_regularization_losses
layer_metrics
`trainable_variables
a	variables
 layer_regularization_losses
metrics
Ă__call__
+Ä&call_and_return_all_conditional_losses
'Ä"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
-0
.1"
trackable_list_wrapper
.
-0
.1"
trackable_list_wrapper
ľ
non_trainable_variables
layers
dregularization_losses
layer_metrics
etrainable_variables
f	variables
 layer_regularization_losses
metrics
Ĺ__call__
+Ć&call_and_return_all_conditional_losses
'Ć"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
ľ
non_trainable_variables
 layers
iregularization_losses
Ąlayer_metrics
jtrainable_variables
k	variables
 ˘layer_regularization_losses
Łmetrics
Ç__call__
+Č&call_and_return_all_conditional_losses
'Č"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
/0
01"
trackable_list_wrapper
.
/0
01"
trackable_list_wrapper
ľ
¤non_trainable_variables
Ľlayers
nregularization_losses
Ślayer_metrics
otrainable_variables
p	variables
 §layer_regularization_losses
¨metrics
É__call__
+Ę&call_and_return_all_conditional_losses
'Ę"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
C
0
1
2
3
4"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Ř2Ő
&__inference_critic_layer_call_fn_54562
&__inference_critic_layer_call_fn_54599
&__inference_critic_layer_call_fn_54787
&__inference_critic_layer_call_fn_54750˛
Š˛Ľ
FullArgSpec(
args 
jself
jinput

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
Ú2×
 __inference__wrapped_model_53524˛
˛
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *"˘

input_1˙˙˙˙˙˙˙˙˙
Ä2Á
A__inference_critic_layer_call_and_return_conditional_losses_54713
A__inference_critic_layer_call_and_return_conditional_losses_54525
A__inference_critic_layer_call_and_return_conditional_losses_54475
A__inference_critic_layer_call_and_return_conditional_losses_54663˛
Š˛Ľ
FullArgSpec(
args 
jself
jinput

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
×2Ô
__inference_call_43875
__inference_call_43925Ą
˛
FullArgSpec
args
jself
jinput
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
×2Ô
(__inference_generate_readout_input_43946§
˛
FullArgSpec"
args
jself
jlink_states
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
Ń2Î
%__inference_message_aggregation_43958¤
˛
FullArgSpec
args
jself

jmessages
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
í2ę
!__inference_message_passing_44574
!__inference_message_passing_44266Ą
˛
FullArgSpec
args
jself
jinput
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
2
.__inference_create_message_layer_call_fn_53652
.__inference_create_message_layer_call_fn_54849
.__inference_create_message_layer_call_fn_54836
.__inference_create_message_layer_call_fn_53625Ŕ
ˇ˛ł
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsŞ 
annotationsŞ *
 
ň2ď
I__inference_create_message_layer_call_and_return_conditional_losses_54805
I__inference_create_message_layer_call_and_return_conditional_losses_54823
I__inference_create_message_layer_call_and_return_conditional_losses_53583
I__inference_create_message_layer_call_and_return_conditional_losses_53597Ŕ
ˇ˛ł
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsŞ 
annotationsŞ *
 
ú2÷
+__inference_link_update_layer_call_fn_54916
+__inference_link_update_layer_call_fn_54933
+__inference_link_update_layer_call_fn_53794
+__inference_link_update_layer_call_fn_53830Ŕ
ˇ˛ł
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsŞ 
annotationsŞ *
 
ć2ă
F__inference_link_update_layer_call_and_return_conditional_losses_53738
F__inference_link_update_layer_call_and_return_conditional_losses_54874
F__inference_link_update_layer_call_and_return_conditional_losses_54899
F__inference_link_update_layer_call_and_return_conditional_losses_53757Ŕ
ˇ˛ł
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsŞ 
annotationsŞ *
 
ź2š
'__inference_readout_layer_call_fn_55033
'__inference_readout_layer_call_fn_55133
'__inference_readout_layer_call_fn_54035
'__inference_readout_layer_call_fn_55116
'__inference_readout_layer_call_fn_55016
'__inference_readout_layer_call_fn_54073Ŕ
ˇ˛ł
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsŞ 
annotationsŞ *
 
Ţ2Ű
B__inference_readout_layer_call_and_return_conditional_losses_53975
B__inference_readout_layer_call_and_return_conditional_losses_55099
B__inference_readout_layer_call_and_return_conditional_losses_55073
B__inference_readout_layer_call_and_return_conditional_losses_54999
B__inference_readout_layer_call_and_return_conditional_losses_54973
B__inference_readout_layer_call_and_return_conditional_losses_53996Ŕ
ˇ˛ł
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsŞ 
annotationsŞ *
 
2B0
#__inference_signature_wrapper_54411input_1
Ń2Î
'__inference_dense_8_layer_call_fn_55153˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
ě2é
B__inference_dense_8_layer_call_and_return_conditional_losses_55144˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
Ń2Î
'__inference_dense_9_layer_call_fn_55173˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
ě2é
B__inference_dense_9_layer_call_and_return_conditional_losses_55164˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
Ň2Ď
(__inference_dense_10_layer_call_fn_55193˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
í2ę
C__inference_dense_10_layer_call_and_return_conditional_losses_55184˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
Ň2Ď
(__inference_dense_11_layer_call_fn_55213˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
í2ę
C__inference_dense_11_layer_call_and_return_conditional_losses_55204˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
Ň2Ď
(__inference_dense_12_layer_call_fn_55233˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
í2ę
C__inference_dense_12_layer_call_and_return_conditional_losses_55224˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
Ň2Ď
(__inference_dense_13_layer_call_fn_55253˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
í2ę
C__inference_dense_13_layer_call_and_return_conditional_losses_55244˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
2
)__inference_dropout_2_layer_call_fn_55275
)__inference_dropout_2_layer_call_fn_55280´
Ť˛§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsŞ 
annotationsŞ *
 
Ć2Ă
D__inference_dropout_2_layer_call_and_return_conditional_losses_55270
D__inference_dropout_2_layer_call_and_return_conditional_losses_55265´
Ť˛§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsŞ 
annotationsŞ *
 
Ň2Ď
(__inference_dense_14_layer_call_fn_55300˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
í2ę
C__inference_dense_14_layer_call_and_return_conditional_losses_55291˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
2
)__inference_dropout_3_layer_call_fn_55327
)__inference_dropout_3_layer_call_fn_55322´
Ť˛§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsŞ 
annotationsŞ *
 
Ć2Ă
D__inference_dropout_3_layer_call_and_return_conditional_losses_55317
D__inference_dropout_3_layer_call_and_return_conditional_losses_55312´
Ť˛§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsŞ 
annotationsŞ *
 
Ň2Ď
(__inference_dense_15_layer_call_fn_55346˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
í2ę
C__inference_dense_15_layer_call_and_return_conditional_losses_55337˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
 __inference__wrapped_model_53524h!"#$%&'()*+,-./0,˘)
"˘

input_1˙˙˙˙˙˙˙˙˙
Ş "&Ş#
!
output_1
output_1]
__inference_call_43875C!"#$%&'()*+,-./0"˘
˘

input
Ş "e
__inference_call_43925K!"#$%&'()*+,-./0*˘'
 ˘

input˙˙˙˙˙˙˙˙˙
Ş "ş
I__inference_create_message_layer_call_and_return_conditional_losses_53583m!"#$>˘;
4˘1
'$
dense_8_input˙˙˙˙˙˙˙˙˙ 
p

 
Ş "%˘"

0˙˙˙˙˙˙˙˙˙
 ş
I__inference_create_message_layer_call_and_return_conditional_losses_53597m!"#$>˘;
4˘1
'$
dense_8_input˙˙˙˙˙˙˙˙˙ 
p 

 
Ş "%˘"

0˙˙˙˙˙˙˙˙˙
 ł
I__inference_create_message_layer_call_and_return_conditional_losses_54805f!"#$7˘4
-˘*
 
inputs˙˙˙˙˙˙˙˙˙ 
p

 
Ş "%˘"

0˙˙˙˙˙˙˙˙˙
 ł
I__inference_create_message_layer_call_and_return_conditional_losses_54823f!"#$7˘4
-˘*
 
inputs˙˙˙˙˙˙˙˙˙ 
p 

 
Ş "%˘"

0˙˙˙˙˙˙˙˙˙
 
.__inference_create_message_layer_call_fn_53625`!"#$>˘;
4˘1
'$
dense_8_input˙˙˙˙˙˙˙˙˙ 
p

 
Ş "˙˙˙˙˙˙˙˙˙
.__inference_create_message_layer_call_fn_53652`!"#$>˘;
4˘1
'$
dense_8_input˙˙˙˙˙˙˙˙˙ 
p 

 
Ş "˙˙˙˙˙˙˙˙˙
.__inference_create_message_layer_call_fn_54836Y!"#$7˘4
-˘*
 
inputs˙˙˙˙˙˙˙˙˙ 
p

 
Ş "˙˙˙˙˙˙˙˙˙
.__inference_create_message_layer_call_fn_54849Y!"#$7˘4
-˘*
 
inputs˙˙˙˙˙˙˙˙˙ 
p 

 
Ş "˙˙˙˙˙˙˙˙˙Ł
A__inference_critic_layer_call_and_return_conditional_losses_54475^!"#$%&'()*+,-./00˘-
&˘#

input_1˙˙˙˙˙˙˙˙˙
p
Ş "˘

0
 Ł
A__inference_critic_layer_call_and_return_conditional_losses_54525^!"#$%&'()*+,-./00˘-
&˘#

input_1˙˙˙˙˙˙˙˙˙
p 
Ş "˘

0
 Ą
A__inference_critic_layer_call_and_return_conditional_losses_54663\!"#$%&'()*+,-./0.˘+
$˘!

input˙˙˙˙˙˙˙˙˙
p
Ş "˘

0
 Ą
A__inference_critic_layer_call_and_return_conditional_losses_54713\!"#$%&'()*+,-./0.˘+
$˘!

input˙˙˙˙˙˙˙˙˙
p 
Ş "˘

0
 {
&__inference_critic_layer_call_fn_54562Q!"#$%&'()*+,-./00˘-
&˘#

input_1˙˙˙˙˙˙˙˙˙
p
Ş "{
&__inference_critic_layer_call_fn_54599Q!"#$%&'()*+,-./00˘-
&˘#

input_1˙˙˙˙˙˙˙˙˙
p 
Ş "y
&__inference_critic_layer_call_fn_54750O!"#$%&'()*+,-./0.˘+
$˘!

input˙˙˙˙˙˙˙˙˙
p
Ş "y
&__inference_critic_layer_call_fn_54787O!"#$%&'()*+,-./0.˘+
$˘!

input˙˙˙˙˙˙˙˙˙
p 
Ş "¤
C__inference_dense_10_layer_call_and_return_conditional_losses_55184]%&/˘,
%˘"
 
inputs˙˙˙˙˙˙˙˙˙0
Ş "&˘#

0˙˙˙˙˙˙˙˙˙
 |
(__inference_dense_10_layer_call_fn_55193P%&/˘,
%˘"
 
inputs˙˙˙˙˙˙˙˙˙0
Ş "˙˙˙˙˙˙˙˙˙¤
C__inference_dense_11_layer_call_and_return_conditional_losses_55204]'(0˘-
&˘#
!
inputs˙˙˙˙˙˙˙˙˙
Ş "%˘"

0˙˙˙˙˙˙˙˙˙@
 |
(__inference_dense_11_layer_call_fn_55213P'(0˘-
&˘#
!
inputs˙˙˙˙˙˙˙˙˙
Ş "˙˙˙˙˙˙˙˙˙@Ł
C__inference_dense_12_layer_call_and_return_conditional_losses_55224\)*/˘,
%˘"
 
inputs˙˙˙˙˙˙˙˙˙@
Ş "%˘"

0˙˙˙˙˙˙˙˙˙
 {
(__inference_dense_12_layer_call_fn_55233O)*/˘,
%˘"
 
inputs˙˙˙˙˙˙˙˙˙@
Ş "˙˙˙˙˙˙˙˙˙¤
C__inference_dense_13_layer_call_and_return_conditional_losses_55244]+,/˘,
%˘"
 
inputs˙˙˙˙˙˙˙˙˙@
Ş "&˘#

0˙˙˙˙˙˙˙˙˙
 |
(__inference_dense_13_layer_call_fn_55253P+,/˘,
%˘"
 
inputs˙˙˙˙˙˙˙˙˙@
Ş "˙˙˙˙˙˙˙˙˙¤
C__inference_dense_14_layer_call_and_return_conditional_losses_55291]-.0˘-
&˘#
!
inputs˙˙˙˙˙˙˙˙˙
Ş "%˘"

0˙˙˙˙˙˙˙˙˙@
 |
(__inference_dense_14_layer_call_fn_55300P-.0˘-
&˘#
!
inputs˙˙˙˙˙˙˙˙˙
Ş "˙˙˙˙˙˙˙˙˙@Ł
C__inference_dense_15_layer_call_and_return_conditional_losses_55337\/0/˘,
%˘"
 
inputs˙˙˙˙˙˙˙˙˙@
Ş "%˘"

0˙˙˙˙˙˙˙˙˙
 {
(__inference_dense_15_layer_call_fn_55346O/0/˘,
%˘"
 
inputs˙˙˙˙˙˙˙˙˙@
Ş "˙˙˙˙˙˙˙˙˙˘
B__inference_dense_8_layer_call_and_return_conditional_losses_55144\!"/˘,
%˘"
 
inputs˙˙˙˙˙˙˙˙˙ 
Ş "%˘"

0˙˙˙˙˙˙˙˙˙@
 z
'__inference_dense_8_layer_call_fn_55153O!"/˘,
%˘"
 
inputs˙˙˙˙˙˙˙˙˙ 
Ş "˙˙˙˙˙˙˙˙˙@˘
B__inference_dense_9_layer_call_and_return_conditional_losses_55164\#$/˘,
%˘"
 
inputs˙˙˙˙˙˙˙˙˙@
Ş "%˘"

0˙˙˙˙˙˙˙˙˙
 z
'__inference_dense_9_layer_call_fn_55173O#$/˘,
%˘"
 
inputs˙˙˙˙˙˙˙˙˙@
Ş "˙˙˙˙˙˙˙˙˙Ś
D__inference_dropout_2_layer_call_and_return_conditional_losses_55265^4˘1
*˘'
!
inputs˙˙˙˙˙˙˙˙˙
p
Ş "&˘#

0˙˙˙˙˙˙˙˙˙
 Ś
D__inference_dropout_2_layer_call_and_return_conditional_losses_55270^4˘1
*˘'
!
inputs˙˙˙˙˙˙˙˙˙
p 
Ş "&˘#

0˙˙˙˙˙˙˙˙˙
 ~
)__inference_dropout_2_layer_call_fn_55275Q4˘1
*˘'
!
inputs˙˙˙˙˙˙˙˙˙
p
Ş "˙˙˙˙˙˙˙˙˙~
)__inference_dropout_2_layer_call_fn_55280Q4˘1
*˘'
!
inputs˙˙˙˙˙˙˙˙˙
p 
Ş "˙˙˙˙˙˙˙˙˙¤
D__inference_dropout_3_layer_call_and_return_conditional_losses_55312\3˘0
)˘&
 
inputs˙˙˙˙˙˙˙˙˙@
p
Ş "%˘"

0˙˙˙˙˙˙˙˙˙@
 ¤
D__inference_dropout_3_layer_call_and_return_conditional_losses_55317\3˘0
)˘&
 
inputs˙˙˙˙˙˙˙˙˙@
p 
Ş "%˘"

0˙˙˙˙˙˙˙˙˙@
 |
)__inference_dropout_3_layer_call_fn_55322O3˘0
)˘&
 
inputs˙˙˙˙˙˙˙˙˙@
p
Ş "˙˙˙˙˙˙˙˙˙@|
)__inference_dropout_3_layer_call_fn_55327O3˘0
)˘&
 
inputs˙˙˙˙˙˙˙˙˙@
p 
Ş "˙˙˙˙˙˙˙˙˙@j
(__inference_generate_readout_input_43946>+˘(
!˘

link_statesJ
Ş "@ş
F__inference_link_update_layer_call_and_return_conditional_losses_53738p%&'()*?˘<
5˘2
(%
dense_10_input˙˙˙˙˙˙˙˙˙0
p

 
Ş "%˘"

0˙˙˙˙˙˙˙˙˙
 ş
F__inference_link_update_layer_call_and_return_conditional_losses_53757p%&'()*?˘<
5˘2
(%
dense_10_input˙˙˙˙˙˙˙˙˙0
p 

 
Ş "%˘"

0˙˙˙˙˙˙˙˙˙
 ˛
F__inference_link_update_layer_call_and_return_conditional_losses_54874h%&'()*7˘4
-˘*
 
inputs˙˙˙˙˙˙˙˙˙0
p

 
Ş "%˘"

0˙˙˙˙˙˙˙˙˙
 ˛
F__inference_link_update_layer_call_and_return_conditional_losses_54899h%&'()*7˘4
-˘*
 
inputs˙˙˙˙˙˙˙˙˙0
p 

 
Ş "%˘"

0˙˙˙˙˙˙˙˙˙
 
+__inference_link_update_layer_call_fn_53794c%&'()*?˘<
5˘2
(%
dense_10_input˙˙˙˙˙˙˙˙˙0
p

 
Ş "˙˙˙˙˙˙˙˙˙
+__inference_link_update_layer_call_fn_53830c%&'()*?˘<
5˘2
(%
dense_10_input˙˙˙˙˙˙˙˙˙0
p 

 
Ş "˙˙˙˙˙˙˙˙˙
+__inference_link_update_layer_call_fn_54916[%&'()*7˘4
-˘*
 
inputs˙˙˙˙˙˙˙˙˙0
p

 
Ş "˙˙˙˙˙˙˙˙˙
+__inference_link_update_layer_call_fn_54933[%&'()*7˘4
-˘*
 
inputs˙˙˙˙˙˙˙˙˙0
p 

 
Ş "˙˙˙˙˙˙˙˙˙e
%__inference_message_aggregation_43958<)˘&
˘

messages	
Ş "J n
!__inference_message_passing_44266I
!"#$%&'()**˘'
 ˘

input˙˙˙˙˙˙˙˙˙
Ş "Jf
!__inference_message_passing_44574A
!"#$%&'()*"˘
˘

input
Ş "Jś
B__inference_readout_layer_call_and_return_conditional_losses_53975p+,-./0?˘<
5˘2
(%
dense_13_input˙˙˙˙˙˙˙˙˙@
p

 
Ş "%˘"

0˙˙˙˙˙˙˙˙˙
 ś
B__inference_readout_layer_call_and_return_conditional_losses_53996p+,-./0?˘<
5˘2
(%
dense_13_input˙˙˙˙˙˙˙˙˙@
p 

 
Ş "%˘"

0˙˙˙˙˙˙˙˙˙
 
B__inference_readout_layer_call_and_return_conditional_losses_54973V+,-./0.˘+
$˘!

inputs@
p

 
Ş "˘

0
 
B__inference_readout_layer_call_and_return_conditional_losses_54999V+,-./0.˘+
$˘!

inputs@
p 

 
Ş "˘

0
 Ž
B__inference_readout_layer_call_and_return_conditional_losses_55073h+,-./07˘4
-˘*
 
inputs˙˙˙˙˙˙˙˙˙@
p

 
Ş "%˘"

0˙˙˙˙˙˙˙˙˙
 Ž
B__inference_readout_layer_call_and_return_conditional_losses_55099h+,-./07˘4
-˘*
 
inputs˙˙˙˙˙˙˙˙˙@
p 

 
Ş "%˘"

0˙˙˙˙˙˙˙˙˙
 
'__inference_readout_layer_call_fn_54035c+,-./0?˘<
5˘2
(%
dense_13_input˙˙˙˙˙˙˙˙˙@
p

 
Ş "˙˙˙˙˙˙˙˙˙
'__inference_readout_layer_call_fn_54073c+,-./0?˘<
5˘2
(%
dense_13_input˙˙˙˙˙˙˙˙˙@
p 

 
Ş "˙˙˙˙˙˙˙˙˙t
'__inference_readout_layer_call_fn_55016I+,-./0.˘+
$˘!

inputs@
p

 
Ş "t
'__inference_readout_layer_call_fn_55033I+,-./0.˘+
$˘!

inputs@
p 

 
Ş "
'__inference_readout_layer_call_fn_55116[+,-./07˘4
-˘*
 
inputs˙˙˙˙˙˙˙˙˙@
p

 
Ş "˙˙˙˙˙˙˙˙˙
'__inference_readout_layer_call_fn_55133[+,-./07˘4
-˘*
 
inputs˙˙˙˙˙˙˙˙˙@
p 

 
Ş "˙˙˙˙˙˙˙˙˙
#__inference_signature_wrapper_54411s!"#$%&'()*+,-./07˘4
˘ 
-Ş*
(
input_1
input_1˙˙˙˙˙˙˙˙˙"&Ş#
!
output_1
output_1