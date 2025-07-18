#version 300 es
// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

layout(location = 0) in vec4 aPosition;
layout(location = 1) in vec2 aTexCoord;

out vec2 vTexCoord;

void main() {
  // Rotate texture 90 degrees clockwise
  vTexCoord = vec2(1.0f - aTexCoord.t, aTexCoord.s);
  gl_Position = aPosition;
}
