�	�*���f@�*���f@!�*���f@	��^Z���?��^Z���?!��^Z���?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6�*���f@82����@1��se@A�HP��?I���v!@YKvl�u�?*	�����!�@2K
Iterator::Model::Map.�!��u�?!M��ew�V@)]�Fx�?1��vV@:Preprocessing2{
DIterator::Model::Map::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate��镲�?!	�����@)�?Ɯ?1��oi.�
@:Preprocessing2q
:Iterator::Model::Map::ParallelMapV2::Zip[1]::ForeverRepeat������?!դ��2@)a2U0*��?1F\�jj^@:Preprocessing2Z
#Iterator::Model::Map::ParallelMapV29��v���?!�W�����?)9��v���?1�W�����?:Preprocessing2F
Iterator::Model=�U����?!(���W@)_�Q�{?1c�ol��?:Preprocessing2_
(Iterator::Model::Map::ParallelMapV2::Zip�p=
ף�?!��6w�@)�~j�t�x?1X�S��?:Preprocessing2�
TIterator::Model::Map::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice��_�Lu?!Ly7�H��?)��_�Lu?1Ly7�H��?:Preprocessing2}
FIterator::Model::Map::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor����Mbp?!tD\��?)����Mbp?1tD\��?:Preprocessing2k
4Iterator::Model::Map::ParallelMapV2::Zip[0]::FlatMap��d�`T�?!���� @){�G�zd?1�jp�Y"�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 0.2% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.moderate"�3.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9��^Z���?#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	82����@82����@!82����@      ��!       "	��se@��se@!��se@*      ��!       2	�HP��?�HP��?!�HP��?:	���v!@���v!@!���v!@B      ��!       J	Kvl�u�?Kvl�u�?!Kvl�u�?R      ��!       Z	Kvl�u�?Kvl�u�?!Kvl�u�?JGPUY��^Z���?b �"_
>gradient_tape/sequential_1/max_pooling2d_4/MaxPool/MaxPoolGradMaxPoolGrad�ó9Y�?!�ó9Y�?"o
Egradient_tape/sequential_1/batch_normalization_7/FusedBatchNormGradV3FusedBatchNormGradV3�����?!52)`��?"i
?gradient_tape/sequential_1/conv2d_4/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter!��ř�?!��)��]�?"Y
3sequential_1/batch_normalization_7/FusedBatchNormV3FusedBatchNormV3�~0��ȳ?!�v �O�?"X
7gradient_tape/sequential_1/conv2d_4/Sigmoid/SigmoidGradSigmoidGrad�B�����?!�lq�}�?"g
>gradient_tape/sequential_1/conv2d_5/Conv2D/Conv2DBackpropInputConv2DBackpropInputE�oH}%�?!y�y|��?":
sequential_1/conv2d_4/BiasAddBiasAdd�$)��?!$7Q��?"i
?gradient_tape/sequential_1/conv2d_5/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter�E }yI�?!i(����?"A
$sequential_1/max_pooling2d_4/MaxPoolMaxPoolQ{�bI�?!0e�8�?":
sequential_1/conv2d_4/SigmoidSigmoid�+O(�?!�"�W��?Q      Y@Yx��\ @a���h��V@q����E;@y��)�E�S?"�
device�Your program is NOT input-bound because only 0.2% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�3.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).:
Refer to the TF2 Profiler FAQb�27.2723% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 