<?php

declare(strict_types=1);

namespace Tests\Plugins\Embeddings\Support;

use AlfacodeTeam\PhpServicePlatform\Kernel\Ports\HttpClientPort;
use AlfacodeTeam\PhpServicePlatform\Kernel\Ports\HttpClientResponse;
use AlfacodeTeam\PhpServicePlatform\Kernel\Ports\PendingRequestContract;

/**
 * An HttpClientPort that answers what the test queued and RECORDS what it was asked.
 *
 * No socket is ever opened. The recording is the point: the request SHAPE
 * (url, headers, JSON body, timeout, retry) is the contract with a provider, and
 * the only place a test can see it is here.
 */
final class RecordingHttpClient implements HttpClientPort
{
    /** @var list<array{method: string, url: string, options: array<string, mixed>}> */
    public array $calls = [];

    /** @var list<HttpClientResponse|\Throwable> */
    private array $queue = [];

    public function queue(HttpClientResponse|\Throwable ...$answers): self
    {
        foreach ($answers as $a) {
            $this->queue[] = $a;
        }

        return $this;
    }

    /** @param list<list<float>> $vectors in INPUT order; sent back in that order unless $order says otherwise */
    public static function embeddingsResponse(array $vectors, ?array $order = null, int $status = 200): HttpClientResponse
    {
        $data = [];

        foreach ($order ?? array_keys($vectors) as $position => $index) {
            $data[] = ['object' => 'embedding', 'index' => $index, 'embedding' => $vectors[$index]];
        }

        return new HttpClientResponse($status, json_encode(['object' => 'list', 'data' => $data, 'model' => 'x']));
    }

    public function request(string $method, string $url, array $options = []): HttpClientResponse
    {
        $this->calls[] = ['method' => $method, 'url' => $url, 'options' => $options];

        $next = array_shift($this->queue) ?? throw new \LogicException('RecordingHttpClient: no answer queued for this request.');

        if ($next instanceof \Throwable) {
            throw $next;
        }

        return $next;
    }

    public function get(string $url, array $query = []): HttpClientResponse
    {
        return $this->request('GET', $url, ['query' => $query]);
    }

    public function post(string $url, array $data = []): HttpClientResponse
    {
        return $this->request('POST', $url, ['body' => $data]);
    }

    public function put(string $url, array $data = []): HttpClientResponse
    {
        return $this->request('PUT', $url, ['body' => $data]);
    }

    public function patch(string $url, array $data = []): HttpClientResponse
    {
        return $this->request('PATCH', $url, ['body' => $data]);
    }

    public function delete(string $url, array $data = []): HttpClientResponse
    {
        return $this->request('DELETE', $url, ['body' => $data]);
    }

    public function pending(): PendingRequestContract
    {
        throw new \LogicException('This plugin goes through request() so every option is explicit and auditable in one place.');
    }
}
