# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: mediapipe/graphs/iris_tracking/calculators/iris_to_depth_calculator.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from mediapipe.framework import calculator_pb2 as mediapipe_dot_framework_dot_calculator__pb2
try:
  mediapipe_dot_framework_dot_calculator__options__pb2 = mediapipe_dot_framework_dot_calculator__pb2.mediapipe_dot_framework_dot_calculator__options__pb2
except AttributeError:
  mediapipe_dot_framework_dot_calculator__options__pb2 = mediapipe_dot_framework_dot_calculator__pb2.mediapipe.framework.calculator_options_pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='mediapipe/graphs/iris_tracking/calculators/iris_to_depth_calculator.proto',
  package='mediapipe',
  syntax='proto2',
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_pb=b'\nImediapipe/graphs/iris_tracking/calculators/iris_to_depth_calculator.proto\x12\tmediapipe\x1a$mediapipe/framework/calculator.proto\"\xcd\x03\n\x1cIrisToDepthCalculatorOptions\x12!\n\x16left_iris_center_index\x18\x01 \x01(\x05:\x01\x30\x12\x1e\n\x13left_iris_top_index\x18\x02 \x01(\x05:\x01\x32\x12!\n\x16left_iris_bottom_index\x18\x03 \x01(\x05:\x01\x34\x12\x1f\n\x14left_iris_left_index\x18\x04 \x01(\x05:\x01\x33\x12 \n\x15left_iris_right_index\x18\x05 \x01(\x05:\x01\x31\x12\"\n\x17right_iris_center_index\x18\x06 \x01(\x05:\x01\x35\x12\x1f\n\x14right_iris_top_index\x18\x07 \x01(\x05:\x01\x37\x12\"\n\x17right_iris_bottom_index\x18\x08 \x01(\x05:\x01\x39\x12 \n\x15right_iris_left_index\x18\t \x01(\x05:\x01\x36\x12!\n\x16right_iris_right_index\x18\n \x01(\x05:\x01\x38\x32V\n\x03\x65xt\x12\x1c.mediapipe.CalculatorOptions\x18\x8a\xeb\xd7\x90\x01 \x01(\x0b\x32\'.mediapipe.IrisToDepthCalculatorOptions'
  ,
  dependencies=[mediapipe_dot_framework_dot_calculator__pb2.DESCRIPTOR,])




_IRISTODEPTHCALCULATOROPTIONS = _descriptor.Descriptor(
  name='IrisToDepthCalculatorOptions',
  full_name='mediapipe.IrisToDepthCalculatorOptions',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='left_iris_center_index', full_name='mediapipe.IrisToDepthCalculatorOptions.left_iris_center_index', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='left_iris_top_index', full_name='mediapipe.IrisToDepthCalculatorOptions.left_iris_top_index', index=1,
      number=2, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=2,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='left_iris_bottom_index', full_name='mediapipe.IrisToDepthCalculatorOptions.left_iris_bottom_index', index=2,
      number=3, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=4,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='left_iris_left_index', full_name='mediapipe.IrisToDepthCalculatorOptions.left_iris_left_index', index=3,
      number=4, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=3,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='left_iris_right_index', full_name='mediapipe.IrisToDepthCalculatorOptions.left_iris_right_index', index=4,
      number=5, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=1,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='right_iris_center_index', full_name='mediapipe.IrisToDepthCalculatorOptions.right_iris_center_index', index=5,
      number=6, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=5,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='right_iris_top_index', full_name='mediapipe.IrisToDepthCalculatorOptions.right_iris_top_index', index=6,
      number=7, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=7,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='right_iris_bottom_index', full_name='mediapipe.IrisToDepthCalculatorOptions.right_iris_bottom_index', index=7,
      number=8, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=9,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='right_iris_left_index', full_name='mediapipe.IrisToDepthCalculatorOptions.right_iris_left_index', index=8,
      number=9, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=6,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='right_iris_right_index', full_name='mediapipe.IrisToDepthCalculatorOptions.right_iris_right_index', index=9,
      number=10, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=8,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
    _descriptor.FieldDescriptor(
      name='ext', full_name='mediapipe.IrisToDepthCalculatorOptions.ext', index=0,
      number=303429002, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=True, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=127,
  serialized_end=588,
)

DESCRIPTOR.message_types_by_name['IrisToDepthCalculatorOptions'] = _IRISTODEPTHCALCULATOROPTIONS
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

IrisToDepthCalculatorOptions = _reflection.GeneratedProtocolMessageType('IrisToDepthCalculatorOptions', (_message.Message,), {
  'DESCRIPTOR' : _IRISTODEPTHCALCULATOROPTIONS,
  '__module__' : 'mediapipe.graphs.iris_tracking.calculators.iris_to_depth_calculator_pb2'
  # @@protoc_insertion_point(class_scope:mediapipe.IrisToDepthCalculatorOptions)
  })
_sym_db.RegisterMessage(IrisToDepthCalculatorOptions)

_IRISTODEPTHCALCULATOROPTIONS.extensions_by_name['ext'].message_type = _IRISTODEPTHCALCULATOROPTIONS
mediapipe_dot_framework_dot_calculator__options__pb2.CalculatorOptions.RegisterExtension(_IRISTODEPTHCALCULATOROPTIONS.extensions_by_name['ext'])

# @@protoc_insertion_point(module_scope)
