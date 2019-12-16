# coding=utf-8
# Copyright 2019 Google LLC
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: gfootball/eval_server/proto/game_server.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='gfootball/eval_server/proto/game_server.proto',
  package='gfootball.eval_server',
  syntax='proto3',
  serialized_options=None,
  serialized_pb=_b('\n-gfootball/eval_server/proto/game_server.proto\x12\x15gfootball.eval_server\"q\n\x13GetEnvResultRequest\x12\x14\n\x0cgame_version\x18\x01 \x01(\t\x12\x0f\n\x07game_id\x18\x02 \x01(\t\x12\x10\n\x08username\x18\x03 \x01(\t\x12\r\n\x05token\x18\x04 \x01(\t\x12\x12\n\nmodel_name\x18\x05 \x01(\t\"*\n\x14GetEnvResultResponse\x12\x12\n\nenv_result\x18\x01 \x01(\x0c\"\x8e\x01\n\x0bStepRequest\x12\x14\n\x0cgame_version\x18\x01 \x01(\t\x12\x0f\n\x07game_id\x18\x02 \x01(\t\x12\x10\n\x08username\x18\x03 \x01(\t\x12\r\n\x05token\x18\x04 \x01(\t\x12\x0e\n\x06\x61\x63tion\x18\x05 \x01(\x05\x12\x12\n\nmodel_name\x18\x06 \x01(\t\x12\x13\n\x0b\x61\x63tion_list\x18\x07 \x03(\x05\"\"\n\x0cStepResponse\x12\x12\n\nenv_result\x18\x01 \x01(\x0c\"\x14\n\x12GetCapacityRequest\"1\n\x13GetCapacityResponse\x12\x1a\n\x12\x63\x61pacity_for_games\x18\x01 \x01(\x05\"y\n\x11\x43reateGameRequest\x12\x0f\n\x07game_id\x18\x01 \x01(\t\x12\x13\n\x0bleft_player\x18\x02 \x01(\t\x12\x14\n\x0cright_player\x18\x03 \x01(\t\x12\x19\n\x11include_rendering\x18\x04 \x01(\x08\x12\r\n\x05track\x18\x05 \x01(\t\"\x14\n\x12\x43reateGameResponse2\x97\x03\n\nGameServer\x12i\n\x0cGetEnvResult\x12*.gfootball.eval_server.GetEnvResultRequest\x1a+.gfootball.eval_server.GetEnvResultResponse\"\x00\x12Q\n\x04Step\x12\".gfootball.eval_server.StepRequest\x1a#.gfootball.eval_server.StepResponse\"\x00\x12\x66\n\x0bGetCapacity\x12).gfootball.eval_server.GetCapacityRequest\x1a*.gfootball.eval_server.GetCapacityResponse\"\x00\x12\x63\n\nCreateGame\x12(.gfootball.eval_server.CreateGameRequest\x1a).gfootball.eval_server.CreateGameResponse\"\x00\x62\x06proto3')
)




_GETENVRESULTREQUEST = _descriptor.Descriptor(
  name='GetEnvResultRequest',
  full_name='gfootball.eval_server.GetEnvResultRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='game_version', full_name='gfootball.eval_server.GetEnvResultRequest.game_version', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='game_id', full_name='gfootball.eval_server.GetEnvResultRequest.game_id', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='username', full_name='gfootball.eval_server.GetEnvResultRequest.username', index=2,
      number=3, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='token', full_name='gfootball.eval_server.GetEnvResultRequest.token', index=3,
      number=4, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='model_name', full_name='gfootball.eval_server.GetEnvResultRequest.model_name', index=4,
      number=5, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=72,
  serialized_end=185,
)


_GETENVRESULTRESPONSE = _descriptor.Descriptor(
  name='GetEnvResultResponse',
  full_name='gfootball.eval_server.GetEnvResultResponse',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='env_result', full_name='gfootball.eval_server.GetEnvResultResponse.env_result', index=0,
      number=1, type=12, cpp_type=9, label=1,
      has_default_value=False, default_value=_b(""),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=187,
  serialized_end=229,
)


_STEPREQUEST = _descriptor.Descriptor(
  name='StepRequest',
  full_name='gfootball.eval_server.StepRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='game_version', full_name='gfootball.eval_server.StepRequest.game_version', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='game_id', full_name='gfootball.eval_server.StepRequest.game_id', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='username', full_name='gfootball.eval_server.StepRequest.username', index=2,
      number=3, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='token', full_name='gfootball.eval_server.StepRequest.token', index=3,
      number=4, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='action', full_name='gfootball.eval_server.StepRequest.action', index=4,
      number=5, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='model_name', full_name='gfootball.eval_server.StepRequest.model_name', index=5,
      number=6, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='action_list', full_name='gfootball.eval_server.StepRequest.action_list', index=6,
      number=7, type=5, cpp_type=1, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=232,
  serialized_end=374,
)


_STEPRESPONSE = _descriptor.Descriptor(
  name='StepResponse',
  full_name='gfootball.eval_server.StepResponse',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='env_result', full_name='gfootball.eval_server.StepResponse.env_result', index=0,
      number=1, type=12, cpp_type=9, label=1,
      has_default_value=False, default_value=_b(""),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=376,
  serialized_end=410,
)


_GETCAPACITYREQUEST = _descriptor.Descriptor(
  name='GetCapacityRequest',
  full_name='gfootball.eval_server.GetCapacityRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=412,
  serialized_end=432,
)


_GETCAPACITYRESPONSE = _descriptor.Descriptor(
  name='GetCapacityResponse',
  full_name='gfootball.eval_server.GetCapacityResponse',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='capacity_for_games', full_name='gfootball.eval_server.GetCapacityResponse.capacity_for_games', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=434,
  serialized_end=483,
)


_CREATEGAMEREQUEST = _descriptor.Descriptor(
  name='CreateGameRequest',
  full_name='gfootball.eval_server.CreateGameRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='game_id', full_name='gfootball.eval_server.CreateGameRequest.game_id', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='left_player', full_name='gfootball.eval_server.CreateGameRequest.left_player', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='right_player', full_name='gfootball.eval_server.CreateGameRequest.right_player', index=2,
      number=3, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='include_rendering', full_name='gfootball.eval_server.CreateGameRequest.include_rendering', index=3,
      number=4, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='track', full_name='gfootball.eval_server.CreateGameRequest.track', index=4,
      number=5, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=485,
  serialized_end=606,
)


_CREATEGAMERESPONSE = _descriptor.Descriptor(
  name='CreateGameResponse',
  full_name='gfootball.eval_server.CreateGameResponse',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=608,
  serialized_end=628,
)

DESCRIPTOR.message_types_by_name['GetEnvResultRequest'] = _GETENVRESULTREQUEST
DESCRIPTOR.message_types_by_name['GetEnvResultResponse'] = _GETENVRESULTRESPONSE
DESCRIPTOR.message_types_by_name['StepRequest'] = _STEPREQUEST
DESCRIPTOR.message_types_by_name['StepResponse'] = _STEPRESPONSE
DESCRIPTOR.message_types_by_name['GetCapacityRequest'] = _GETCAPACITYREQUEST
DESCRIPTOR.message_types_by_name['GetCapacityResponse'] = _GETCAPACITYRESPONSE
DESCRIPTOR.message_types_by_name['CreateGameRequest'] = _CREATEGAMEREQUEST
DESCRIPTOR.message_types_by_name['CreateGameResponse'] = _CREATEGAMERESPONSE
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

GetEnvResultRequest = _reflection.GeneratedProtocolMessageType('GetEnvResultRequest', (_message.Message,), {
  'DESCRIPTOR' : _GETENVRESULTREQUEST,
  '__module__' : 'gfootball.eval_server.proto.game_server_pb2'
  # @@protoc_insertion_point(class_scope:gfootball.eval_server.GetEnvResultRequest)
  })
_sym_db.RegisterMessage(GetEnvResultRequest)

GetEnvResultResponse = _reflection.GeneratedProtocolMessageType('GetEnvResultResponse', (_message.Message,), {
  'DESCRIPTOR' : _GETENVRESULTRESPONSE,
  '__module__' : 'gfootball.eval_server.proto.game_server_pb2'
  # @@protoc_insertion_point(class_scope:gfootball.eval_server.GetEnvResultResponse)
  })
_sym_db.RegisterMessage(GetEnvResultResponse)

StepRequest = _reflection.GeneratedProtocolMessageType('StepRequest', (_message.Message,), {
  'DESCRIPTOR' : _STEPREQUEST,
  '__module__' : 'gfootball.eval_server.proto.game_server_pb2'
  # @@protoc_insertion_point(class_scope:gfootball.eval_server.StepRequest)
  })
_sym_db.RegisterMessage(StepRequest)

StepResponse = _reflection.GeneratedProtocolMessageType('StepResponse', (_message.Message,), {
  'DESCRIPTOR' : _STEPRESPONSE,
  '__module__' : 'gfootball.eval_server.proto.game_server_pb2'
  # @@protoc_insertion_point(class_scope:gfootball.eval_server.StepResponse)
  })
_sym_db.RegisterMessage(StepResponse)

GetCapacityRequest = _reflection.GeneratedProtocolMessageType('GetCapacityRequest', (_message.Message,), {
  'DESCRIPTOR' : _GETCAPACITYREQUEST,
  '__module__' : 'gfootball.eval_server.proto.game_server_pb2'
  # @@protoc_insertion_point(class_scope:gfootball.eval_server.GetCapacityRequest)
  })
_sym_db.RegisterMessage(GetCapacityRequest)

GetCapacityResponse = _reflection.GeneratedProtocolMessageType('GetCapacityResponse', (_message.Message,), {
  'DESCRIPTOR' : _GETCAPACITYRESPONSE,
  '__module__' : 'gfootball.eval_server.proto.game_server_pb2'
  # @@protoc_insertion_point(class_scope:gfootball.eval_server.GetCapacityResponse)
  })
_sym_db.RegisterMessage(GetCapacityResponse)

CreateGameRequest = _reflection.GeneratedProtocolMessageType('CreateGameRequest', (_message.Message,), {
  'DESCRIPTOR' : _CREATEGAMEREQUEST,
  '__module__' : 'gfootball.eval_server.proto.game_server_pb2'
  # @@protoc_insertion_point(class_scope:gfootball.eval_server.CreateGameRequest)
  })
_sym_db.RegisterMessage(CreateGameRequest)

CreateGameResponse = _reflection.GeneratedProtocolMessageType('CreateGameResponse', (_message.Message,), {
  'DESCRIPTOR' : _CREATEGAMERESPONSE,
  '__module__' : 'gfootball.eval_server.proto.game_server_pb2'
  # @@protoc_insertion_point(class_scope:gfootball.eval_server.CreateGameResponse)
  })
_sym_db.RegisterMessage(CreateGameResponse)



_GAMESERVER = _descriptor.ServiceDescriptor(
  name='GameServer',
  full_name='gfootball.eval_server.GameServer',
  file=DESCRIPTOR,
  index=0,
  serialized_options=None,
  serialized_start=631,
  serialized_end=1038,
  methods=[
  _descriptor.MethodDescriptor(
    name='GetEnvResult',
    full_name='gfootball.eval_server.GameServer.GetEnvResult',
    index=0,
    containing_service=None,
    input_type=_GETENVRESULTREQUEST,
    output_type=_GETENVRESULTRESPONSE,
    serialized_options=None,
  ),
  _descriptor.MethodDescriptor(
    name='Step',
    full_name='gfootball.eval_server.GameServer.Step',
    index=1,
    containing_service=None,
    input_type=_STEPREQUEST,
    output_type=_STEPRESPONSE,
    serialized_options=None,
  ),
  _descriptor.MethodDescriptor(
    name='GetCapacity',
    full_name='gfootball.eval_server.GameServer.GetCapacity',
    index=2,
    containing_service=None,
    input_type=_GETCAPACITYREQUEST,
    output_type=_GETCAPACITYRESPONSE,
    serialized_options=None,
  ),
  _descriptor.MethodDescriptor(
    name='CreateGame',
    full_name='gfootball.eval_server.GameServer.CreateGame',
    index=3,
    containing_service=None,
    input_type=_CREATEGAMEREQUEST,
    output_type=_CREATEGAMERESPONSE,
    serialized_options=None,
  ),
])
_sym_db.RegisterServiceDescriptor(_GAMESERVER)

DESCRIPTOR.services_by_name['GameServer'] = _GAMESERVER

# @@protoc_insertion_point(module_scope)
