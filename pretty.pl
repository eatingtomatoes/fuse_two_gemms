#! /usr/bin/env perl

use strict;
use warnings;

# 描述模板参数的结构(嵌套的括号结构)
my $type_args = qr{
   (?<bracket>
       <
            (?:
                (?> [^<>]+ )
                | (?&bracket)
            )*
       >
   )
}x;

sub format_layout {
    return $1
        # 把 shape 与 stride 之间的逗号换成冒号
        =~ s/(?<shape>cute::tuple$type_args),\s*?(?<stride>cute::tuple$type_args)/$+{shape}:$+{stride}/gr

        # 把 `tuple<...>` 简写为 `<...>`
        =~ s/cute::tuple$type_args/$1/ger

        # 把 `C<数字>` 简写为 `数字`
        =~ s/C<(\d+)>/$1/gr

        # 把 `<` 改为 `(`
        =~ s/</\(/gr

        # 把 `>` 改为 `)`
        =~ s/>/\)/gr

        # 删掉所有 `cute::` 前缀
        =~ s/cute:://gr

        # 把 `int` 改写为 `dyn`
        =~ s/int/dyn/gr;
}

while (<>) {
    print s/cute::Layout$type_args/'layout' . format_layout($1)/ger;
}
