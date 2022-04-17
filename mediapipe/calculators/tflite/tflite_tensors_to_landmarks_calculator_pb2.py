# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: mediapipe/calculators/tflite/tflite_tensors_to_landmarks_calculator.proto
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
  name='mediapipe/calculators/tflite/tflite_tensors_to_landmarks_calculator.proto',
  package='mediapipe',
  syntax='proto2',
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_pb=b'\nImediapipe/calculators/tflite/tflite_tensors_to_landmarks_calculator.proto\x12\tmediapipe\x1a$mediapipe/framework/calculator.proto\"\xa6\x04\n)TfLiteTensorsToLandmarksCalculatorOptions\x12\x15\n\rnum_landmarks\x18\x01 \x02(\x05\x12\x19\n\x11input_image_width\x18\x02 \x01(\x05\x12\x1a\n\x12input_image_height\x18\x03 \x01(\x05\x12\x1e\n\x0f\x66lip_vertically\x18\x04 \x01(\x08:\x05\x66\x61lse\x12 \n\x11\x66lip_horizontally\x18\x06 \x01(\x08:\x05\x66\x61lse\x12\x16\n\x0bnormalize_z\x18\x05 \x01(\x02:\x01\x31\x12\x64\n\x15visibility_activation\x18\x07 \x01(\x0e\x32?.mediapipe.TfLiteTensorsToLandmarksCalculatorOptions.Activation:\x04NONE\x12\x62\n\x13presence_activation\x18\x08 \x01(\x0e\x32?.mediapipe.TfLiteTensorsToLandmarksCalculatorOptions.Activation:\x04NONE\"#\n\nActivation\x12\x08\n\x04NONE\x10\x00\x12\x0b\n\x07SIGMOID\x10\x01\x32\x62\n\x03\x65xt\x12\x1c.mediapipe.CalculatorOptions\x18\xca\xe0\xdez \x01(\x0b\x32\x34.mediapipe.TfLiteTensorsToLandmarksCalculatorOptions'
  ,
  dependencies=[mediapipe_dot_framework_dot_calculator__pb2.DESCRIPTOR,])



_TFLITETENSORSTOLANDMARKSCALCULATOROPTIONS_ACTIVATION = _descriptor.EnumDescriptor(
  name='Activation',
  full_name='mediapipe.TfLiteTensorsToLandmarksCalculatorOptions.Activation',
  filename=None,
  file=DESCRIPTOR,
  create_key=_descriptor._internal_create_key,
  values=[
    _descriptor.EnumValueDescriptor(
      name='NONE', index=0, number=0,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='SIGMOID', index=1, number=1,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=542,
  serialized_end=577,
)
_sym_db.RegisterEnumDescriptor(_TFLITETENSORSTOLANDMARKSCALCULATOROPTIONS_ACTIVATION)


_TFLITETENSORSTOLANDMARKSCALCULATOROPTIONS = _descriptor.Descriptor(
  name='TfLiteTensorsToLandmarksCalculatorOptions',
  full_name='mediapipe.TfLiteTensorsToLandmarksCalculatorOptions',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='num_landmarks', full_name='mediapipe.TfLiteTensorsToLandmarksCalculatorOptions.num_landmarks', index=0,
      number=1, type=5, cpp_type=1, label=2,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='input_image_width', full_name='mediapipe.TfLiteTensorsToLandmarksCalculatorOptions.input_image_width', index=1,
      number=2, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='input_image_height', full_name='mediapipe.TfLiteTensorsToLandmarksCalculatorOptions.input_image_height', index=2,
      number=3, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='flip_vertically', full_name='mediapipe.TfLiteTensorsToLandmarksCalculatorOptions.flip_vertically', index=3,
      number=4, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='flip_horizontally', full_name='mediapipe.TfLiteTensorsToLandmarksCalculatorOptions.flip_horizontally', index=4,
      number=6, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='normalize_z', full_name='mediapipe.TfLiteTensorsToLandmarksCalculatorOptions.normalize_z', index=5,
      number=5, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(1),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='visibility_activation', full_name='mediapipe.TfLiteTensorsToLandmarksCalculatorOptions.visibility_activation', index=6,
      number=7, type=14, cpp_type=8, label=1,
      has_default_value=True, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='presence_activation', full_name='mediapipe.TfLiteTensorsToLandmarksCalculatorOptions.presence_activation', index=7,
      number=8, type=14, cpp_type=8, label=1,
      has_default_value=True, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
    _descriptor.FieldDescriptor(
      name='ext', full_name='mediapipe.TfLiteTensorsToLandmarksCalculatorOptions.ext', index=0,
      number=257405002, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=True, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  nested_types=[],
  enum_types=[
    _TFLITETENSORSTOLANDMARKSCALCULATOROPTIONS_ACTIVATION,
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=127,
  serialized_end=677,
)

_TFLITETENSORSTOLANDMARKSCALCULATOROPTIONS.fields_by_name['visibility_activation'].enum_type = _TFLITETENSORSTOLANDMARKSCALCULATOROPTIONS_ACTIVATION
_TFLITETENSORSTOLANDMARKSCALCULATOROPTIONS.fields_by_name['presence_activation'].enum_type = _TFLITETENSORSTOLANDMARKSCALCULATOROPTIONS_ACTIVATION
_TFLITETENSORSTOLANDMARKSCALCULATOROPTIONS_ACTIVATION.containing_type = _TFLITETENSORSTOLANDMARKSCALCULATOROPTIONS
DESCRIPTOR.message_types_by_name['TfLiteTensorsToLandmarksCalculatorOptions'] = _TFLITETENSORSTOLANDMARKSCALCULATOROPTIONS
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

TfLiteTensorsToLandmarksCalculatorOptions = _reflection.GeneratedProtocolMessageType('TfLiteTensorsToLandmarksCalculatorOptions', (_message.Message,), {
  'DESCRIPTOR' : _TFLITETENSORSTOLANDMARKSCALCULATOROPTIONS,
  '__module__' : 'mediapipe.calculators.tflite.tflite_tensors_to_landmarks_calculator_pb2'
  # @@protoc_insertion_point(class_scope:mediapipe.TfLiteTensorsToLandmarksCalculatorOptions)
  })
_sym_db.RegisterMessage(TfLiteTensorsToLandmarksCalculatorOptions)

_TFLITETENSORSTOLANDMARKSCALCULATOROPTIONS.extensions_by_name['ext'].message_type = _TFLITETENSORSTOLANDMARKSCALCULATOROPTIONS
mediapipe_dot_framework_dot_calculator__options__pb2.CalculatorOptions.RegisterExtension(_TFLITETENSORSTOLANDMARKSCALCULATOROPTIONS.extensions_by_name['ext'])

# @@protoc_insertion_point(module_scope)
