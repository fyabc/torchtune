# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from typing import List

from torchtune.data import ChatMLFormat, Message


class Qwen2ChatFormat(ChatMLFormat):
    """Qwen2 chat format.

    Main differences between this chat format and :class:`~torchtune.data._chat_formats.ChatMLFormat`:

        - formatted assistant message has a trailing "\n"
        - if the last message has role assistant and empty content, will use `assistant_for_generation`
            instead of `assistant` template (for generation).
    """

    IM_START, IM_END = "<|im_start|>", "<|im_end|>"
    system = f"{IM_START}system\n{{content}}{IM_END}\n"
    user = f"{IM_START}user\n{{content}}{IM_END}\n"
    assistant = f"{IM_START}assistant\n{{content}}{IM_END}\n"
    assistant_for_generation = f"{IM_START}assistant\n"

    @classmethod
    def format(
        cls,
        sample: List[Message],
    ) -> List[Message]:
        """
        Format user and system messages with appropriate tags.

        Args:
            sample (List[Message]): a single conversation, structured as a list
                of `Message` objects

        Returns:
            The formatted list of messages
        """
        formatted_dialogue = []
        for index, message in enumerate(sample):
            content = ""
            if message.role == "system":
                content = cls.system.format(content=message.content)
            elif message.role == "user":
                content = cls.user.format(
                    content=message.content,
                )
            elif message.role == "assistant":
                if index == len(sample) - 1 and not message.content:
                    content = cls.assistant_for_generation
                else:
                    content = cls.assistant.format(
                        content=message.content,
                    )
            formatted_dialogue.append(
                Message(role=message.role, content=content, masked=message.masked),
            )
        return formatted_dialogue