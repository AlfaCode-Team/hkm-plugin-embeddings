<?php

declare(strict_types=1);

namespace Tests\Plugins\Embeddings\Unit;

use AlfacodeTeam\PhpServicePlatform\Kernel\Ports\HttpClientResponse;
use AlfacodeTeam\PhpServicePlatform\Kernel\Ports\Vector\EmbeddingException;
use PHPUnit\Framework\TestCase;
use Plugins\Embeddings\Infrastructure\Config\EmbeddingsConfig;
use Plugins\Embeddings\Infrastructure\Gateways\OpenAiCompatibleEmbedder;
use Tests\Plugins\Embeddings\Support\RecordingHttpClient;

/**
 * NO LIVE PROVIDER WAS CALLED. Every test here drives the adapter through a
 * recording HttpClientPort, so what is verified is the request this class SENDS
 * and how it reads a canned answer. The request/response shapes are OpenAI's
 * documented embeddings API as recalled by the author; none was checked against
 * a live endpoint.
 */
final class OpenAiCompatibleEmbedderTest extends TestCase
{
    private const KEY = 'sk-test-0123456789abcdef-SECRET';

    private static function config(array $override = []): EmbeddingsConfig
    {
        return EmbeddingsConfig::of(...($override + [
            'driver'     => EmbeddingsConfig::DRIVER_OPENAI,
            'model'      => 'text-embedding-3-small',
            'dimensions' => 3,
            'apiKey'     => self::KEY,
        ]));
    }

    public function test_the_exact_request_for_one_text(): void
    {
        $http = (new RecordingHttpClient())->queue(RecordingHttpClient::embeddingsResponse([[0.1, 0.2, 0.3]]));

        (new OpenAiCompatibleEmbedder($http, self::config()))->embed('hello');

        self::assertCount(1, $http->calls);
        $call = $http->calls[0];

        self::assertSame('POST', $call['method']);
        self::assertSame('https://api.openai.com/v1/embeddings', $call['url']);
        self::assertSame(
            ['model' => 'text-embedding-3-small', 'input' => ['hello'], 'encoding_format' => 'float', 'dimensions' => 3],
            $call['options']['json'],
        );
        self::assertSame(['Authorization' => 'Bearer ' . self::KEY, 'Accept' => 'application/json'], $call['options']['headers']);
        self::assertSame(30, $call['options']['timeout']);
        self::assertSame(10, $call['options']['connect_timeout']);
        self::assertSame(0, $call['options']['retry'], 'one attempt: the caller owns the decision to retry a paid call');
        self::assertSame(1 * 3 * 24 + 65536, $call['options']['max_response_bytes']);
    }

    public function test_dimensions_is_sent_only_when_the_operator_set_it(): void
    {
        $http = (new RecordingHttpClient())->queue(RecordingHttpClient::embeddingsResponse([array_fill(0, 1536, 0.5)]));

        (new OpenAiCompatibleEmbedder($http, self::config(['dimensions' => null])))->embed('x');

        self::assertArrayNotHasKey('dimensions', $http->calls[0]['options']['json'], 'a model that does not take the field would reject it');
    }

    public function test_the_base_url_is_normalised_and_configurable(): void
    {
        $http = (new RecordingHttpClient())->queue(RecordingHttpClient::embeddingsResponse([[1, 2, 3]]));

        (new OpenAiCompatibleEmbedder($http, self::config(['baseUrl' => 'http://localhost:11434/v1/'])))->embed('x');

        self::assertSame('http://localhost:11434/v1/embeddings', $http->calls[0]['url']);
    }

    public function test_batches_are_split_by_batch_size_and_results_keep_input_order(): void
    {
        $http = (new RecordingHttpClient())->queue(
            RecordingHttpClient::embeddingsResponse([[1, 0, 0], [0, 1, 0]]),
            RecordingHttpClient::embeddingsResponse([[0, 0, 1], [1, 1, 0]]),
            RecordingHttpClient::embeddingsResponse([[0, 1, 1]]),
        );

        $out = (new OpenAiCompatibleEmbedder($http, self::config(['batchSize' => 2])))->embedMany(['a', 'b', 'c', 'd', 'e']);

        self::assertCount(3, $http->calls, 'ceil(5 / 2) requests, never one per text');
        self::assertSame([['a', 'b'], ['c', 'd'], ['e']], array_map(static fn (array $c): array => $c['options']['json']['input'], $http->calls));
        self::assertSame(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 0.0], [0.0, 1.0, 1.0]],
            array_map(static fn ($v): array => $v->toArray(), $out),
        );
    }

    public function test_the_providers_index_decides_the_order_not_the_array_position(): void
    {
        // The provider answers out of order: position 0 carries index 1 and vice versa.
        $http = (new RecordingHttpClient())->queue(RecordingHttpClient::embeddingsResponse([[1, 0, 0], [0, 1, 0]], order: [1, 0]));

        $out = (new OpenAiCompatibleEmbedder($http, self::config()))->embedMany(['first', 'second']);

        self::assertSame([1.0, 0.0, 0.0], $out[0]->toArray());
        self::assertSame([0.0, 1.0, 0.0], $out[1]->toArray());
    }

    public function test_an_empty_list_asks_nobody(): void
    {
        $http = new RecordingHttpClient();

        self::assertSame([], (new OpenAiCompatibleEmbedder($http, self::config()))->embedMany([]));
        self::assertSame([], $http->calls);
    }

    public function test_blank_and_oversized_input_is_refused_before_any_request(): void
    {
        $http = new RecordingHttpClient();
        $e    = new OpenAiCompatibleEmbedder($http, self::config(['maxInputChars' => 10]));

        foreach (['', '   ', str_repeat('x', 11)] as $bad) {
            try {
                $e->embed($bad);
                self::fail('refused input must not reach the provider');
            } catch (\InvalidArgumentException) {
            }
        }

        self::assertSame([], $http->calls, 'a refused input is free');
    }

    public function test_no_api_key_fails_before_any_request(): void
    {
        $http = new RecordingHttpClient();

        try {
            (new OpenAiCompatibleEmbedder($http, self::config(['apiKey' => ''])))->embed('x');
            self::fail('no key, no call');
        } catch (EmbeddingException $e) {
            self::assertStringContainsString('EMBEDDINGS_API_KEY is not set', $e->getMessage());
        }

        self::assertSame([], $http->calls);
    }

    /** @return iterable<string, array{HttpClientResponse, string}> */
    public static function malformed(): iterable
    {
        yield 'wrong count'      => [RecordingHttpClient::embeddingsResponse([[1, 2, 3]]), 'returned 1 vectors for 2 inputs'];
        yield 'no data key'      => [new HttpClientResponse(200, '{"object":"list"}'), 'no "data" list for 2 inputs'];
        yield 'not json'         => [new HttpClientResponse(200, '<html>gateway</html>'), 'no "data" list'];
        yield 'duplicate index'  => [new HttpClientResponse(200, '{"data":[{"index":0,"embedding":[1,2,3]},{"index":0,"embedding":[1,2,3]}]}'), 'malformed or duplicate "index"'];
        yield 'index out of range' => [new HttpClientResponse(200, '{"data":[{"index":0,"embedding":[1,2,3]},{"index":5,"embedding":[1,2,3]}]}'), 'malformed or duplicate "index"'];
        yield 'string index'     => [new HttpClientResponse(200, '{"data":[{"index":"0","embedding":[1,2,3]},{"index":1,"embedding":[1,2,3]}]}'), 'malformed or duplicate "index"'];
        yield 'missing embedding' => [new HttpClientResponse(200, '{"data":[{"index":0},{"index":1,"embedding":[1,2,3]}]}'), 'malformed or duplicate "index"'];
        yield 'base64 string embedding' => [new HttpClientResponse(200, '{"data":[{"index":0,"embedding":"AAAA"},{"index":1,"embedding":"AAAA"}]}'), 'malformed or duplicate "index"'];
        yield 'wrong dimension'  => [RecordingHttpClient::embeddingsResponse([[1, 2], [3, 4]]), 'returned a 2-dimension vector; 3 were expected'];
        yield 'non-numeric value' => [new HttpClientResponse(200, '{"data":[{"index":0,"embedding":[1,"x",3]},{"index":1,"embedding":[1,2,3]}]}'), 'unusable vector'];
    }

    #[\PHPUnit\Framework\Attributes\DataProvider('malformed')]
    public function test_a_malformed_answer_is_an_error_never_a_wrong_vector(HttpClientResponse $answer, string $reason): void
    {
        $http = (new RecordingHttpClient())->queue($answer);

        try {
            (new OpenAiCompatibleEmbedder($http, self::config()))->embedMany(['a', 'b']);
            self::fail('a malformed response must not produce vectors');
        } catch (EmbeddingException $e) {
            self::assertStringContainsString($reason, $e->getMessage());
            self::assertSame(200, $e->status);
        }
    }

    public function test_a_401_that_echoes_the_key_does_not_put_the_providers_text_in_the_exception(): void
    {
        // OpenAI's 401 quotes a masked copy of the key it was sent. Here the "provider" echoes the whole thing.
        $http = (new RecordingHttpClient())->queue(new HttpClientResponse(
            401,
            json_encode(['error' => ['message' => 'Incorrect API key provided: ' . self::KEY, 'type' => 'invalid_request_error', 'code' => 'invalid_api_key']]),
        ));

        try {
            (new OpenAiCompatibleEmbedder($http, self::config()))->embed('x');
            self::fail('401 must throw');
        } catch (EmbeddingException $e) {
            self::assertSame(401, $e->status);
            self::assertSame('The embedding provider answered HTTP 401 (invalid_api_key).', $e->getMessage());
            self::assertStringNotContainsString(self::KEY, $e->getMessage());
            self::assertStringNotContainsString('Incorrect', $e->getMessage());
        }
    }

    public function test_a_hostile_error_code_is_not_copied(): void
    {
        $http = (new RecordingHttpClient())->queue(new HttpClientResponse(
            500,
            json_encode(['error' => ['code' => "x\nInjected: line " . self::KEY]]),
        ));

        try {
            (new OpenAiCompatibleEmbedder($http, self::config()))->embed('x');
            self::fail();
        } catch (EmbeddingException $e) {
            self::assertSame('The embedding provider answered HTTP 500.', $e->getMessage());
        }
    }

    public function test_a_429_keeps_its_code_and_status(): void
    {
        $http = (new RecordingHttpClient())->queue(new HttpClientResponse(429, '{"error":{"code":"rate_limit_exceeded","message":"slow down"}}'));

        try {
            (new OpenAiCompatibleEmbedder($http, self::config()))->embed('x');
            self::fail();
        } catch (EmbeddingException $e) {
            self::assertSame(429, $e->status);
            self::assertStringContainsString('(rate_limit_exceeded)', $e->getMessage());
            self::assertStringNotContainsString('slow down', $e->getMessage());
        }
    }

    public function test_a_transport_failure_keeps_the_cause_but_not_its_text(): void
    {
        $cause = new \RuntimeException('cURL error 28: timed out connecting to https://api.openai.com/v1/embeddings with header Authorization: Bearer ' . self::KEY);
        $http  = (new RecordingHttpClient())->queue($cause);

        try {
            (new OpenAiCompatibleEmbedder($http, self::config()))->embed('x');
            self::fail();
        } catch (EmbeddingException $e) {
            self::assertSame($cause, $e->getPrevious(), 'the cause stays available to a debugger');
            self::assertStringNotContainsString(self::KEY, $e->getMessage());
            self::assertStringNotContainsString('api.openai.com', $e->getMessage());
            self::assertStringContainsString('RuntimeException', $e->getMessage());
        }
    }

    public function test_the_key_is_in_the_real_request_and_nowhere_else(): void
    {
        $http = (new RecordingHttpClient())->queue(RecordingHttpClient::embeddingsResponse([[1, 2, 3]]));
        $e    = new OpenAiCompatibleEmbedder($http, self::config());

        $e->embed('x');

        $real = json_encode($http->calls[0]);
        self::assertStringContainsString(self::KEY, $real, 'the real request carries the key — that is the point of it');

        $shown = json_encode($e->describeRequest(['x']));
        self::assertStringNotContainsString(self::KEY, $shown);
        self::assertStringContainsString('Bearer ***', $shown);

        self::assertStringNotContainsString(self::KEY, $e->model());
        self::assertStringNotContainsString(self::KEY, print_r(self::config(), true));
        self::assertStringNotContainsString(self::KEY, (string) json_encode(self::config()), 'nor a json_encode of it');
        self::assertStringNotContainsString(self::KEY, var_export(self::config(), true), 'nor a var_export');
        ob_start();
        var_dump(self::config());
        self::assertStringNotContainsString(self::KEY, (string) ob_get_clean(), 'a var_dump of the config must not show the key');
    }
}
