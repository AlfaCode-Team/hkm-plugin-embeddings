<?php

declare(strict_types=1);

namespace Plugins\Embeddings\Infrastructure\Gateways;

use AlfacodeTeam\PhpServicePlatform\Kernel\Ports\EmbeddingPort;
use AlfacodeTeam\PhpServicePlatform\Kernel\Ports\HttpClientPort;
use AlfacodeTeam\PhpServicePlatform\Kernel\Ports\Vector\EmbeddingException;
use AlfacodeTeam\PhpServicePlatform\Kernel\Ports\Vector\Vector;
use Plugins\Embeddings\Domain\EmbeddingInput;
use Plugins\Embeddings\Infrastructure\Config\EmbeddingsConfig;

/**
 * EmbeddingPort over any OpenAI-compatible `POST {base}/embeddings` endpoint,
 * through HttpClientPort — never raw cURL.
 *
 * ════════════════════════════════════════════════════════════════════════════
 * NO LIVE PROVIDER WAS CALLED
 * ════════════════════════════════════════════════════════════════════════════
 * Everything below is verified against a RECORDING HttpClientPort: the exact
 * request this class sends is asserted, and canned responses are read back. The
 * request and response SHAPES come from OpenAI's published embeddings API as the
 * author recalls it — `{"model", "input": [...], "encoding_format", "dimensions"?}`
 * out, `{"data": [{"index", "embedding": [...]}, ...]}` back — and were NOT
 * checked against a live endpoint, nor against a compatible server's
 * documentation. A provider that deviates (a different field name, base64 by
 * default) will surface as an EmbeddingException on the first call, never as a
 * wrong vector: every response is validated for count, order and length.
 *
 * ════════════════════════════════════════════════════════════════════════════
 * THE CREDENTIAL
 * ════════════════════════════════════════════════════════════════════════════
 * Held in {@see EmbeddingsConfig}, read once per batch to build the
 * Authorization header, and nowhere else. It is not in any exception message and
 * not in {@see self::model()}. Provider error TEXT is not copied into an
 * exception either: OpenAI's 401 message quotes a masked copy of the key it was
 * sent ("Incorrect API key provided: sk-abc****xyz", recalled), and a copied
 * message ends up in logs. Only the HTTP status and a short machine `code` are
 * kept.
 *
 * ════════════════════════════════════════════════════════════════════════════
 * ONE ATTEMPT
 * ════════════════════════════════════════════════════════════════════════════
 * `retry => 0`. A POST is not in the HTTP client's default retry set, and the
 * decision to retry a paid call, and how often, belongs to the caller (a queue
 * job has its own retry policy). The cache makes a retried batch cheap: the
 * inputs that did succeed are already stored.
 */
final class OpenAiCompatibleEmbedder implements EmbeddingPort
{
    public function __construct(
        private readonly HttpClientPort $http,
        private readonly EmbeddingsConfig $config,
    ) {}

    public function embed(string $text): Vector
    {
        return $this->embedMany([$text])[0];
    }

    public function embedMany(array $texts): array
    {
        EmbeddingInput::assertTexts($texts, $this->config->maxInputChars);

        if ($texts === []) {
            return [];
        }

        if (!$this->config->hasApiKey()) {
            throw new EmbeddingException('EMBEDDINGS_API_KEY is not set, so the openai driver cannot authenticate.', 'embed');
        }

        $out = [];

        foreach (array_chunk($texts, $this->config->batchSize) as $batch) {
            foreach ($this->batch($batch) as $vector) {
                $out[] = $vector;
            }
        }

        return $out;
    }

    public function dimensions(): int
    {
        return $this->config->dimensions;
    }

    public function model(): string
    {
        return $this->config->model;
    }

    /**
     * The request this adapter sends for a batch, with the credential masked.
     * For `embeddings:probe` and for tests; building it here keeps the real
     * request and the displayed one from drifting apart.
     *
     * @param list<string> $batch
     * @return array{method: string, url: string, headers: array<string, string>, json: array<string, mixed>}
     */
    public function describeRequest(array $batch): array
    {
        $request                          = $this->buildRequest($batch);
        $request['headers']['Authorization'] = 'Bearer ***';

        return ['method' => 'POST', 'url' => $request['url'], 'headers' => $request['headers'], 'json' => $request['json']];
    }

    /**
     * @param list<string> $batch
     * @return array{url: string, headers: array<string, string>, json: array<string, mixed>}
     */
    private function buildRequest(array $batch): array
    {
        $json = [
            'model'           => $this->config->model,
            'input'           => $batch,
            'encoding_format' => 'float',
        ];

        // Only when asked for: a model that does not take `dimensions` (older ones) would
        // reject the field, and one that does has a default that the config already mirrors.
        if ($this->config->dimensionsExplicit) {
            $json['dimensions'] = $this->config->dimensions;
        }

        return [
            'url'     => $this->config->baseUrl . '/embeddings',
            'headers' => [
                'Authorization' => 'Bearer ' . $this->config->revealApiKey(),
                'Accept'        => 'application/json',
            ],
            'json'    => $json,
        ];
    }

    /**
     * @param list<string> $batch
     * @return list<Vector>
     */
    private function batch(array $batch): array
    {
        $request = $this->buildRequest($batch);

        try {
            $response = $this->http->request('POST', $request['url'], [
                'headers'         => $request['headers'],
                'json'            => $request['json'],
                'timeout'         => $this->config->timeout,
                'connect_timeout' => min($this->config->timeout, 10),
                'retry'           => 0,
                // A float32 is ~12 bytes of JSON; 24 leaves headroom for longer digits, plus the envelope.
                'max_response_bytes' => \count($batch) * $this->config->dimensions * 24 + 65536,
            ]);
        } catch (EmbeddingException $e) {
            throw $e;
        } catch (\Throwable $e) {
            // Transport failure. The exception text from an HTTP client can carry the URL and, in
            // some clients, request headers — so it is kept as `previous` and not copied here.
            throw new EmbeddingException(
                'The embedding request could not be completed (' . $e::class . ').',
                'embed',
                null,
                $e,
            );
        }

        if (!$response->ok()) {
            throw new EmbeddingException(
                \sprintf('The embedding provider answered HTTP %d%s.', $response->status(), self::errorCode($response->json())),
                'embed',
                $response->status(),
            );
        }

        return $this->read($response->json(), \count($batch), $response->status());
    }

    /**
     * @return list<Vector>
     */
    private function read(mixed $body, int $expected, int $status): array
    {
        $data = \is_array($body) ? ($body['data'] ?? null) : null;

        if (!\is_array($data) || \count($data) !== $expected) {
            throw new EmbeddingException(
                \sprintf('The embedding provider returned %s for %d inputs.', \is_array($data) ? \count($data) . ' vectors' : 'no "data" list', $expected),
                'embed',
                $status,
            );
        }

        // Providers state each vector's input position in `index`; do not trust the ARRAY order.
        $byIndex = [];

        foreach ($data as $item) {
            $index = \is_array($item) ? ($item['index'] ?? null) : null;

            if (!\is_int($index) || $index < 0 || $index >= $expected || isset($byIndex[$index]) || !\is_array($item['embedding'] ?? null)) {
                throw new EmbeddingException('The embedding provider returned a malformed or duplicate "index".', 'embed', $status);
            }

            $byIndex[$index] = $item['embedding'];
        }

        $out = [];

        for ($i = 0; $i < $expected; $i++) {
            $values = $byIndex[$i];

            if (\count($values) !== $this->config->dimensions) {
                throw new EmbeddingException(
                    \sprintf(
                        'The embedding provider returned a %d-dimension vector; %d were expected (EMBEDDINGS_DIMENSIONS / the model\'s own size). It was not padded or truncated: a vector of the wrong length is a different model.',
                        \count($values),
                        $this->config->dimensions,
                    ),
                    'embed',
                    $status,
                );
            }

            try {
                $out[] = Vector::of($values);
            } catch (\InvalidArgumentException $e) {
                throw new EmbeddingException('The embedding provider returned an unusable vector: ' . $e->getMessage(), 'embed', $status, $e);
            }
        }

        return $out;
    }

    /** " (rate_limit_exceeded)" from a provider error body, or "" — never the provider's prose. */
    private static function errorCode(mixed $body): string
    {
        $code = \is_array($body) && \is_array($body['error'] ?? null) ? ($body['error']['code'] ?? $body['error']['type'] ?? null) : null;

        return \is_string($code) && preg_match('/^[A-Za-z0-9_.-]{1,64}$/D', $code) === 1 ? " ({$code})" : '';
    }
}
