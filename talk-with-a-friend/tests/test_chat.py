import os
from urllib.parse import urlsplit, parse_qs

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

import app as chat


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.delenv('TALK_PUBLIC_URL', raising=False)
    monkeypatch.delenv('RENDER', raising=False)
    chat.rooms.clear()
    chat.creation_rate.clear()
    with TestClient(chat.app) as api:
        yield api
    chat.rooms.clear()
    chat.creation_rate.clear()


def test_room_invite_uses_fragment_not_server_logged_query(client):
    response = client.post('/api/rooms')
    assert response.status_code == 200
    details = response.json()
    assert details['room_id'] and details['host_key']
    link = urlsplit(details['invite_url'])
    assert not link.query
    assert 'invite=' in link.fragment
    assert details['host_key'] not in details['invite_url']
    assert client.get('/api/rooms/' + details['room_id']).json()['persisted'] is False
    assert 'host_key' not in client.get('/api/rooms/' + details['room_id']).text


def test_third_party_token_cannot_join(client):
    room = client.post('/api/rooms').json()
    with client.websocket_connect('/ws/' + room['room_id']) as socket:
        socket.send_json({'type':'join', 'token':'made-up-token-long-enough-to-validate', 'name':'Attacker'})
        assert socket.receive_json()['type'] == 'error'
        with pytest.raises(WebSocketDisconnect):
            socket.receive_json()


def test_two_person_invite_messaging_and_end(client):
    room = client.post('/api/rooms').json()
    room_id = room['room_id']
    guest_token = parse_qs(urlsplit(room['invite_url']).fragment)['invite'][0]
    with client.websocket_connect('/ws/' + room_id) as host:
        host.send_json({'type':'join', 'token':room['host_key'], 'name':'Host'})
        assert host.receive_json()['seat'] == 'host'
        assert host.receive_json()['online'] is False
        with client.websocket_connect('/ws/' + room_id) as guest:
            guest.send_json({'type':'join', 'token':guest_token, 'name':'Friend'})
            assert guest.receive_json()['seat'] == 'guest'
            assert host.receive_json()['online'] is True
            assert guest.receive_json()['online'] is True
            raw_text = '<script>harmless string, never executable</script>'
            host.send_json({'type':'message', 'text':raw_text})
            delivered = guest.receive_json()
            assert delivered['type'] == 'message'
            assert delivered['text'] == raw_text
            assert host.receive_json()['type'] == 'sent'
            guest.send_json({'type':'message', 'text':'I am listening.'})
            assert host.receive_json()['text'] == 'I am listening.'
            assert guest.receive_json()['type'] == 'sent'
            # No message history field or durable storage of message contents.
            assert 'messages' not in vars(chat.rooms[room_id])
            host.send_json({'type':'end'})
            assert guest.receive_json()['type'] == 'ended'
        assert client.get('/api/rooms/' + room_id).status_code == 404


def test_only_two_people_and_limit_message_length(client):
    room = client.post('/api/rooms').json()
    token = parse_qs(urlsplit(room['invite_url']).fragment)['invite'][0]
    with client.websocket_connect('/ws/' + room['room_id']) as host:
        host.send_json({'type':'join', 'token':room['host_key'], 'name':'Host'})
        host.receive_json(); host.receive_json()
        with client.websocket_connect('/ws/' + room['room_id']) as guest:
            guest.send_json({'type':'join', 'token':token, 'name':'Friend'})
            guest.receive_json(); host.receive_json(); guest.receive_json()
            with pytest.raises(WebSocketDisconnect):
                with client.websocket_connect('/ws/' + room['room_id']) as intruder:
                    intruder.send_json({'type':'join', 'token':token, 'name':'Another'})
                    error = intruder.receive_json()
                    assert error['type'] == 'error'
                    intruder.receive_json()
            host.send_json({'type':'message', 'text':'a'*1001})
            assert host.receive_json()['type'] == 'error'


def test_room_rate_limit(client):
    for _ in range(5):
        assert client.post('/api/rooms').status_code == 200
    assert client.post('/api/rooms').status_code == 429


def test_room_origin_rejection(client, monkeypatch):
    monkeypatch.setenv('TALK_PUBLIC_URL', 'https://talk.example.org')
    room_id = list(chat.rooms)[0] if chat.rooms else '0'*14
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect('/ws/' + room_id, headers={'origin':'https://third-party.example'}) as socket:
            socket.receive_json()


def test_static_and_health(client):
    assert client.get('/').status_code == 200
    assert 'Start a conversation' in client.get('/').text
    assert client.get('/healthz').json()['mode'] == 'ephemeral-invite-chat'
    assert client.get('/assets/style.css').status_code == 200
    assert client.get('/assets/app.js').status_code == 200
