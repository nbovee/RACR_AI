# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: colab_vision.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x12\x63olab_vision.proto\"\x15\n\x04\x44ict\x12\r\n\x05\x63hunk\x18\x01 \x01(\x0c\"\x16\n\x05\x43hunk\x12\r\n\x05\x63hunk\x18\x01 \x01(\x0c\"\x12\n\x04uuid\x12\n\n\x02id\x18\x01 \x01(\t\" \n\x10result_Time_Dict\x12\x0c\n\x04\x64ict\x18\x01 \x01(\t\"2\n\rFile_Metadata\x12\x0c\n\x04\x61uth\x18\x01 \x01(\t\x12\x13\n\x0b\x66ile_format\x18\x02 \x01(\t\"\x1f\n\x03\x41\x63k\x12\x0c\n\x04\x63ode\x18\x01 \x01(\x05\x12\n\n\x02id\x18\x02 \x01(\t2\x9d\x01\n\x0c\x63olab_vision\x12\x1c\n\nuploadFile\x12\x06.Chunk\x1a\x04.Ack(\x01\x12\x1e\n\x0buploadImage\x12\x06.Chunk\x1a\x05.Dict(\x01\x12\x1f\n\x0c\x64ownloadFile\x12\x05.uuid\x1a\x06.Chunk0\x01\x12.\n\x12resultTimeDownload\x12\x05.uuid\x1a\x11.result_Time_Dictb\x06proto3')



_DICT = DESCRIPTOR.message_types_by_name['Dict']
_CHUNK = DESCRIPTOR.message_types_by_name['Chunk']
_UUID = DESCRIPTOR.message_types_by_name['uuid']
_RESULT_TIME_DICT = DESCRIPTOR.message_types_by_name['result_Time_Dict']
_FILE_METADATA = DESCRIPTOR.message_types_by_name['File_Metadata']
_ACK = DESCRIPTOR.message_types_by_name['Ack']
Dict = _reflection.GeneratedProtocolMessageType('Dict', (_message.Message,), {
  'DESCRIPTOR' : _DICT,
  '__module__' : 'colab_vision_pb2'
  # @@protoc_insertion_point(class_scope:Dict)
  })
_sym_db.RegisterMessage(Dict)

Chunk = _reflection.GeneratedProtocolMessageType('Chunk', (_message.Message,), {
  'DESCRIPTOR' : _CHUNK,
  '__module__' : 'colab_vision_pb2'
  # @@protoc_insertion_point(class_scope:Chunk)
  })
_sym_db.RegisterMessage(Chunk)

uuid = _reflection.GeneratedProtocolMessageType('uuid', (_message.Message,), {
  'DESCRIPTOR' : _UUID,
  '__module__' : 'colab_vision_pb2'
  # @@protoc_insertion_point(class_scope:uuid)
  })
_sym_db.RegisterMessage(uuid)

result_Time_Dict = _reflection.GeneratedProtocolMessageType('result_Time_Dict', (_message.Message,), {
  'DESCRIPTOR' : _RESULT_TIME_DICT,
  '__module__' : 'colab_vision_pb2'
  # @@protoc_insertion_point(class_scope:result_Time_Dict)
  })
_sym_db.RegisterMessage(result_Time_Dict)

File_Metadata = _reflection.GeneratedProtocolMessageType('File_Metadata', (_message.Message,), {
  'DESCRIPTOR' : _FILE_METADATA,
  '__module__' : 'colab_vision_pb2'
  # @@protoc_insertion_point(class_scope:File_Metadata)
  })
_sym_db.RegisterMessage(File_Metadata)

Ack = _reflection.GeneratedProtocolMessageType('Ack', (_message.Message,), {
  'DESCRIPTOR' : _ACK,
  '__module__' : 'colab_vision_pb2'
  # @@protoc_insertion_point(class_scope:Ack)
  })
_sym_db.RegisterMessage(Ack)

_COLAB_VISION = DESCRIPTOR.services_by_name['colab_vision']
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  _DICT._serialized_start=22
  _DICT._serialized_end=43
  _CHUNK._serialized_start=45
  _CHUNK._serialized_end=67
  _UUID._serialized_start=69
  _UUID._serialized_end=87
  _RESULT_TIME_DICT._serialized_start=89
  _RESULT_TIME_DICT._serialized_end=121
  _FILE_METADATA._serialized_start=123
  _FILE_METADATA._serialized_end=173
  _ACK._serialized_start=175
  _ACK._serialized_end=206
  _COLAB_VISION._serialized_start=209
  _COLAB_VISION._serialized_end=366
# @@protoc_insertion_point(module_scope)