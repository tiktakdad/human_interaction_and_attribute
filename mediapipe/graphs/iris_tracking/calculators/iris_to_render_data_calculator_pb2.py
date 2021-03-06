# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: mediapipe/graphs/iris_tracking/calculators/iris_to_render_data_calculator.proto
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
from mediapipe.util import color_pb2 as mediapipe_dot_util_dot_color__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='mediapipe/graphs/iris_tracking/calculators/iris_to_render_data_calculator.proto',
  package='mediapipe',
  syntax='proto2',
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_pb=b'\nOmediapipe/graphs/iris_tracking/calculators/iris_to_render_data_calculator.proto\x12\tmediapipe\x1a$mediapipe/framework/calculator.proto\x1a\x1amediapipe/util/color.proto\"\xfa\x03\n!IrisToRenderDataCalculatorOptions\x12$\n\noval_color\x18\x01 \x01(\x0b\x32\x10.mediapipe.Color\x12(\n\x0elandmark_color\x18\t \x01(\x0b\x32\x10.mediapipe.Color\x12\x19\n\x0eoval_thickness\x18\x02 \x01(\x01:\x01\x31\x12\x1d\n\x12landmark_thickness\x18\n \x01(\x01:\x01\x31\x12\x1a\n\x0e\x66ont_height_px\x18\x03 \x01(\x05:\x02\x35\x30\x12\x1f\n\x14horizontal_offset_px\x18\x07 \x01(\x05:\x01\x30\x12\x1d\n\x12vertical_offset_px\x18\x08 \x01(\x05:\x01\x30\x12\x14\n\tfont_face\x18\x05 \x01(\x05:\x01\x30\x12Q\n\x08location\x18\x06 \x01(\x0e\x32\x35.mediapipe.IrisToRenderDataCalculatorOptions.Location:\x08TOP_LEFT\")\n\x08Location\x12\x0c\n\x08TOP_LEFT\x10\x00\x12\x0f\n\x0b\x42OTTOM_LEFT\x10\x01\x32[\n\x03\x65xt\x12\x1c.mediapipe.CalculatorOptions\x18\xb8\xc1\x87\x8a\x01 \x01(\x0b\x32,.mediapipe.IrisToRenderDataCalculatorOptions'
  ,
  dependencies=[mediapipe_dot_framework_dot_calculator__pb2.DESCRIPTOR,mediapipe_dot_util_dot_color__pb2.DESCRIPTOR,])



_IRISTORENDERDATACALCULATOROPTIONS_LOCATION = _descriptor.EnumDescriptor(
  name='Location',
  full_name='mediapipe.IrisToRenderDataCalculatorOptions.Location',
  filename=None,
  file=DESCRIPTOR,
  create_key=_descriptor._internal_create_key,
  values=[
    _descriptor.EnumValueDescriptor(
      name='TOP_LEFT', index=0, number=0,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='BOTTOM_LEFT', index=1, number=1,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=533,
  serialized_end=574,
)
_sym_db.RegisterEnumDescriptor(_IRISTORENDERDATACALCULATOROPTIONS_LOCATION)


_IRISTORENDERDATACALCULATOROPTIONS = _descriptor.Descriptor(
  name='IrisToRenderDataCalculatorOptions',
  full_name='mediapipe.IrisToRenderDataCalculatorOptions',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='oval_color', full_name='mediapipe.IrisToRenderDataCalculatorOptions.oval_color', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='landmark_color', full_name='mediapipe.IrisToRenderDataCalculatorOptions.landmark_color', index=1,
      number=9, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='oval_thickness', full_name='mediapipe.IrisToRenderDataCalculatorOptions.oval_thickness', index=2,
      number=2, type=1, cpp_type=5, label=1,
      has_default_value=True, default_value=float(1),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='landmark_thickness', full_name='mediapipe.IrisToRenderDataCalculatorOptions.landmark_thickness', index=3,
      number=10, type=1, cpp_type=5, label=1,
      has_default_value=True, default_value=float(1),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='font_height_px', full_name='mediapipe.IrisToRenderDataCalculatorOptions.font_height_px', index=4,
      number=3, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=50,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='horizontal_offset_px', full_name='mediapipe.IrisToRenderDataCalculatorOptions.horizontal_offset_px', index=5,
      number=7, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='vertical_offset_px', full_name='mediapipe.IrisToRenderDataCalculatorOptions.vertical_offset_px', index=6,
      number=8, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='font_face', full_name='mediapipe.IrisToRenderDataCalculatorOptions.font_face', index=7,
      number=5, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='location', full_name='mediapipe.IrisToRenderDataCalculatorOptions.location', index=8,
      number=6, type=14, cpp_type=8, label=1,
      has_default_value=True, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
    _descriptor.FieldDescriptor(
      name='ext', full_name='mediapipe.IrisToRenderDataCalculatorOptions.ext', index=0,
      number=289530040, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=True, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  nested_types=[],
  enum_types=[
    _IRISTORENDERDATACALCULATOROPTIONS_LOCATION,
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=161,
  serialized_end=667,
)

_IRISTORENDERDATACALCULATOROPTIONS.fields_by_name['oval_color'].message_type = mediapipe_dot_util_dot_color__pb2._COLOR
_IRISTORENDERDATACALCULATOROPTIONS.fields_by_name['landmark_color'].message_type = mediapipe_dot_util_dot_color__pb2._COLOR
_IRISTORENDERDATACALCULATOROPTIONS.fields_by_name['location'].enum_type = _IRISTORENDERDATACALCULATOROPTIONS_LOCATION
_IRISTORENDERDATACALCULATOROPTIONS_LOCATION.containing_type = _IRISTORENDERDATACALCULATOROPTIONS
DESCRIPTOR.message_types_by_name['IrisToRenderDataCalculatorOptions'] = _IRISTORENDERDATACALCULATOROPTIONS
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

IrisToRenderDataCalculatorOptions = _reflection.GeneratedProtocolMessageType('IrisToRenderDataCalculatorOptions', (_message.Message,), {
  'DESCRIPTOR' : _IRISTORENDERDATACALCULATOROPTIONS,
  '__module__' : 'mediapipe.graphs.iris_tracking.calculators.iris_to_render_data_calculator_pb2'
  # @@protoc_insertion_point(class_scope:mediapipe.IrisToRenderDataCalculatorOptions)
  })
_sym_db.RegisterMessage(IrisToRenderDataCalculatorOptions)

_IRISTORENDERDATACALCULATOROPTIONS.extensions_by_name['ext'].message_type = _IRISTORENDERDATACALCULATOROPTIONS
mediapipe_dot_framework_dot_calculator__options__pb2.CalculatorOptions.RegisterExtension(_IRISTORENDERDATACALCULATOROPTIONS.extensions_by_name['ext'])

# @@protoc_insertion_point(module_scope)
