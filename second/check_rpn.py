# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""

.. _l-example-backend-api:

ONNX Runtime Backend for ONNX
=============================

*ONNX Runtime* extends the 
`onnx backend API <https://github.com/onnx/onnx/blob/master/docs/ImplementingAnOnnxBackend.md>`_
to run predictions using this runtime.
Let's use the API to compute the prediction
of a simple logistic regression model.
"""
import numpy as np
import onnxruntime as ort
from onnxruntime import datasets

from onnxruntime.capi.onnxruntime_pybind11_state import InvalidArgument
import pdb
from PIL import Image
# import onnxruntime.backend as backend
# from onnx import load

import fire
import time




def ort_run(config_path=None,
            model_dir=None,
            result_path=None,
            create_folder=False,
            display_step=50,
            summary_step=5,
            pickle_result=True):

    #################################  Session option  ########################################
    sess_options = ort.SessionOptions()

    # #### Set graph optimization level
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    # sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC 
    # sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED  
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    # sess_options.enable_profiling = True

    #### save optimized graph in ONNX IR? 
    #### if possible, load the optimized graph & execute 
    #### then, the optimized graph is saved in ONNX IR
    # sess_options.optimized_model_filepath = "/home/uho/workspace/ONNX_pointpillar/second/disabled_optimization.onnx" 
    # sess_options.optimized_model_filepath = "/home/uho/workspace/ONNX_pointpillar/second/layout_optimized.onnx" 


    # pfe_session = ort.InferenceSession("pfe.onnx", sess_options)
    # pscat_session = ort.InferenceSession("pscat.onnx", sess_options)
    rpn_session = ort.InferenceSession("rpn.onnx", sess_options)
    # nms_session = ort.InferenceSession("nms.onnx", sess_options)

    # pfe_session = ort.InferenceSession("pfe.onnx")
    # pscat_session = ort.InferenceSession("pscat.onnx")
    # rpn_session = ort.InferenceSession("rpn.onnx")
    # nms_session = ort.InferenceSession("nms.onnx")

    # pfe_session.set_providers(  ['CPUExecutionProvider'])
    # pscat_session.set_providers(['CPUExecutionProvider'])
    rpn_session.set_providers(  ['CPUExecutionProvider'])
    # nms_session.set_providers(  ['CPUExecutionProvider'])

    # pfe_session.set_providers(  ['CUDAExecutionProvider'])
    # pscat_session.set_providers(['CUDAExecutionProvider'])
    # rpn_session.set_providers(  ['CUDAExecutionProvider'])
    # nms_session.set_providers(  ['CUDAExecutionProvider'])



    ####  Pillar Feature Network  ####
    # pdb.set_trace()

    # x_0 = np.random.random(pfe_session.get_inputs()[0].shape)
    # x_1 = np.random.random(pfe_session.get_inputs()[1].shape)
    # x_2 = np.random.random(pfe_session.get_inputs()[2].shape)
    # x_3 = np.random.random(pfe_session.get_inputs()[3].shape)
    # x_4 = np.random.random(pfe_session.get_inputs()[4].shape)
    # x_5 = np.random.random(pfe_session.get_inputs()[5].shape)
    # x_6 = np.random.random(pfe_session.get_inputs()[6].shape)
    # x_7 = np.random.random(pfe_session.get_inputs()[7].shape)

    # x_0 = x_0.astype(np.float32)
    # x_1 = x_1.astype(np.float32)
    # x_2 = x_2.astype(np.float32)
    # x_3 = x_3.astype(np.float32)
    # x_4 = x_4.astype(np.float32)
    # x_5 = x_5.astype(np.float32)
    # x_6 = x_6.astype(np.float32)
    # x_7 = x_7.astype(np.float32)


    # pfe_inputs = {pfe_session.get_inputs()[0].name: (x_0),
    #               pfe_session.get_inputs()[1].name: (x_1),
    #               pfe_session.get_inputs()[2].name: (x_2),
    #               pfe_session.get_inputs()[3].name: (x_3),
    #               pfe_session.get_inputs()[4].name: (x_4),
    #               pfe_session.get_inputs()[5].name: (x_5),
    #               pfe_session.get_inputs()[6].name: (x_6),
    #               pfe_session.get_inputs()[7].name: (x_7)
    #               }

    # pfe_interval = 0
    # t = time.time()
    # pfe_outs = pfe_session.run(None, pfe_inputs)
    # pfe_interval += (time.time() - t)

    # print('\n')
    # print('time: ', pfe_interval)
    print('\n')

    ####  Reigion Proposal Network  ####

    input_name = rpn_session.get_inputs()[0].name
    input_shape = rpn_session.get_inputs()[0].shape
    input_type = rpn_session.get_inputs()[0].type

    output_name = rpn_session.get_outputs()[0].name
    output_shape = rpn_session.get_outputs()[0].shape
    output_type = rpn_session.get_outputs()[0].type

    print("Input name  :", input_name)
    print("Input shape :", input_shape)
    print("Input type  :", input_type)
    print("Output name  :", output_name)  
    print("Output shape :", output_shape)
    print("Output type  :", output_type)

    x = np.random.random(input_shape)
    x = x.astype(np.float32)
    
    rpn_interval = 0
    

    t = time.time()
    rpn_outs = rpn_session.run(None, {input_name: x})
    rpn_interval += (time.time() - t)

    print('time: ', rpn_interval)
    print('\n')


    return 0


if __name__ == '__main__':
    ort_run()
    # fire.Fire()