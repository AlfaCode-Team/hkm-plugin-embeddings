<?php

declare(strict_types=1);

namespace Tests\Plugins\Embeddings\Unit;

use AlfacodeTeam\PhpServicePlatform\Kernel\Ports\Vector\Vector;
use PHPUnit\Framework\Attributes\DataProvider;
use PHPUnit\Framework\TestCase;
use Plugins\Embeddings\Infrastructure\Gateways\LocalHashEmbedder;

final class LocalHashEmbedderTest extends TestCase
{
    /** @param list<float> $values @return array<int, float> index => value, non-zero only */
    private static function sparse(array $values): array
    {
        return array_filter($values, static fn (float $x): bool => $x !== 0.0);
    }

    private static function cosine(Vector $a, Vector $b): float
    {
        $dot = 0.0;

        foreach ($a->toArray() as $i => $x) {
            $dot += $x * $b->toArray()[$i];
        }

        return $dot; // both are unit length
    }

    /**
     * Known answer. The expected values were computed by an independent Python
     * implementation of the documented algorithm (sha256, big-endian uint32 of
     * the first four bytes mod dims, sign from the low bit of byte 4, L2 norm),
     * NOT by this class:
     *
     *   tokens of "The quick brown fox, the lazy dog! Éclair 42" are
     *   the quick brown fox the lazy dog éclair 42   — "the" twice
     *   dims=16 → {0:-1/3, 1:1/3, 4:1/3, 6:-1/3, 8:1/3, 13:-2/3}
     */
    public function test_known_answer_against_an_independent_implementation(): void
    {
        $v = (new LocalHashEmbedder(16))->embed('The quick brown fox, the lazy dog! Éclair 42');

        $third = 1 / 3;
        self::assertEqualsWithDelta(
            [0 => -$third, 1 => $third, 4 => $third, 6 => -$third, 8 => $third, 13 => -2 * $third],
            self::sparse($v->toArray()),
            1e-12,
        );
    }

    public function test_a_text_that_is_only_punctuation_falls_back_to_the_whole_text_as_one_token(): void
    {
        // Python: embedding of '!!!' at dims=8 is {6: -1.0}.
        $v = (new LocalHashEmbedder(8))->embed('!!!');

        self::assertEqualsWithDelta([6 => -1.0], self::sparse($v->toArray()), 1e-12);
    }

    public function test_tokens_that_cancel_exactly_fall_back_instead_of_returning_the_zero_vector(): void
    {
        // Found by search in Python: at dims=8, "aa" lands on index 3 with sign -1 and "cd" on index 3
        // with sign +1, so their sum is exactly zero. The fallback (whole text as one token) is {7: -1.0}.
        $v = (new LocalHashEmbedder(8))->embed('aa cd');

        self::assertFalse($v->isZero(), 'a zero vector must never leave the embedder: a cosine collection refuses it');
        self::assertEqualsWithDelta([7 => -1.0], self::sparse($v->toArray()), 1e-12);
    }

    public function test_every_vector_is_unit_length_and_deterministic(): void
    {
        $e = new LocalHashEmbedder(64);

        foreach (['hello world', 'a', 'Ünïcödé テキスト', str_repeat('lorem ipsum ', 100), '1 2 3'] as $text) {
            $a = $e->embed($text);
            $b = (new LocalHashEmbedder(64))->embed($text);

            self::assertSame($a->toArray(), $b->toArray(), 'same text, same vector, in any instance');
            self::assertEqualsWithDelta(1.0, sqrt(array_sum(array_map(static fn (float $x): float => $x * $x, $a->toArray()))), 1e-12);
        }
    }

    /**
     * Independent expectations from Python at dims=256:
     *   cos("red green blue", "blue red green") = 1.0   (a bag of words ignores order)
     *   cos("red green", "red blue")            = 0.5   (one of two words shared)
     *   cos("alpha beta", "gamma delta")        = 0.0   (disjoint, no hash collision at this size)
     */
    public function test_shared_words_give_positive_similarity_and_disjoint_words_none(): void
    {
        $e = new LocalHashEmbedder(256);

        self::assertEqualsWithDelta(1.0, self::cosine($e->embed('red green blue'), $e->embed('blue red green')), 1e-12);
        self::assertEqualsWithDelta(0.5, self::cosine($e->embed('red green'), $e->embed('red blue')), 1e-12);
        self::assertEqualsWithDelta(0.0, self::cosine($e->embed('alpha beta'), $e->embed('gamma delta')), 1e-12);
    }

    public function test_case_is_ignored_and_it_says_so_about_itself(): void
    {
        $e = new LocalHashEmbedder(64);

        self::assertSame($e->embed('Hello World')->toArray(), $e->embed('hello world')->toArray());
        self::assertSame('local-hash-v1', $e->model());
        self::assertSame(64, $e->dimensions());
    }

    public function test_embed_many_keeps_order_and_count(): void
    {
        $e   = new LocalHashEmbedder(32);
        $out = $e->embedMany(['one', 'two', 'three']);

        self::assertCount(3, $out);
        self::assertSame($e->embed('two')->toArray(), $out[1]->toArray());
        self::assertSame([], $e->embedMany([]));
    }

    /** @return iterable<string, array{array<mixed>, string}> */
    public static function badInput(): iterable
    {
        yield 'empty string'      => [[''], 'is blank'];
        yield 'whitespace only'   => [["  \n\t "], 'is blank'];
        yield 'second is blank'   => [['fine', ' '], 'Text 1 is blank'];
        yield 'not a list'        => [['a' => 'x'], 'takes a list'];
        yield 'an int'            => [[5], 'is int, not a string'];
        yield 'null'              => [[null], 'is null, not a string'];
        yield 'over the cap'      => [[str_repeat('x', 51)], 'is 51 characters; the limit is 50'];
    }

    /** @param array<mixed> $texts */
    #[DataProvider('badInput')]
    public function test_bad_input_is_refused_with_the_reason(array $texts, string $reason): void
    {
        $this->expectException(\InvalidArgumentException::class);
        $this->expectExceptionMessageMatches('/' . preg_quote($reason, '/') . '/');

        (new LocalHashEmbedder(16, maxInputChars: 50))->embedMany($texts);
    }

    public function test_the_length_cap_counts_characters_not_bytes(): void
    {
        $e = new LocalHashEmbedder(16, maxInputChars: 5);

        self::assertSame(16, $e->embed('héllo')->dimensions(), '5 characters, 6 bytes: allowed');

        $this->expectException(\InvalidArgumentException::class);
        $e->embed('héllo!');
    }

    public function test_dimensions_are_bounded(): void
    {
        self::assertSame(8, (new LocalHashEmbedder(8))->dimensions());
        self::assertSame(2000, (new LocalHashEmbedder(2000))->dimensions());

        foreach ([7, 0, 2001] as $bad) {
            try {
                new LocalHashEmbedder($bad);
                self::fail("{$bad} dimensions should be refused");
            } catch (\InvalidArgumentException $e) {
                self::assertStringContainsString('between 8 and 2000', $e->getMessage());
            }
        }
    }
}
